from typing import List, Dict, Any, Union, Generator, Optional, AsyncGenerator
import asyncio
from openai import OpenAI  # Changed import
from .classes import BaseLLMProvider, LLMResponse, Usage
from .config import LLMConfig
import logging
from .base_provider import ProviderError, ConfigurationError

# Set up logging
logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models"""
    
    PROVIDER_CONFIG = {
        "client_class": OpenAI,
        "api_key_attr": "openai_api_key",  # Assuming this exists in LLMConfig
        "supports_caching": True # Based on usage details provided
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the OpenAI provider
        
        Args:
            config: LLMConfig instance. If None, will attempt to load from environment
        """
        self.config = config or LLMConfig.from_env()
        
        # Get API key from config
        # The OpenAI client can also load from OPENAI_API_KEY env var if key is None
        api_key = getattr(self.config, self.PROVIDER_CONFIG["api_key_attr"], None)
            
        try:
            # OpenAI client defaults to reading OPENAI_API_KEY from env if api_key is not passed
            self.client = self.PROVIDER_CONFIG["client_class"](api_key=api_key)
            self.supports_caching = self.PROVIDER_CONFIG["supports_caching"]
            self._current_generation = None
            self.last_usage = None
            self.last_response = None
            
            logger.info("Successfully initialized OpenAI provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            raise ConfigurationError(f"Failed to initialize OpenAI provider: {str(e)}")

    async def generate(self, 
                messages: List[Dict[str, Any]], 
                stream: bool = False,
                **kwargs) -> Union[LLMResponse, Generator[str, None, None]]:
        """
        Generate a response using the OpenAI API
        
        Args:
            messages: List of message dictionaries (passed as 'input' to API)
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to the API
        
        Returns:
            Either a LLMResponse object or a Generator for streaming
        """
        try:
            if stream:
                # Streaming needs to be an async generator
                return self._stream_response(messages, **kwargs) 

            # Extract model from kwargs
            model = kwargs.pop('model', None)
            if not model:
                raise ValueError("Model parameter is required")
            
            # Map max_tokens to max_output_tokens if provided
            if 'max_tokens' in kwargs and 'max_output_tokens' not in kwargs:
                 kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
            # Ensure max_output_tokens has a default value if not provided
            if 'max_output_tokens' not in kwargs:
                kwargs['max_output_tokens'] = 1024 # Default like Anthropic's max_tokens
            
            logger.debug(f"Generating response with model: {model}")

            # Strip prefix for API call
            api_model_name = model
            if api_model_name.startswith("responses-"):
                api_model_name = api_model_name[len("responses-"):]
                logger.debug(f"Stripped prefix, using API model name: {api_model_name}")

            request_params = {
                "model": api_model_name, # API uses 'input' and expects stripped model name
                "input": messages, 
                **kwargs
            }
            
            # Assuming the client might be synchronous based on examples
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.responses.create(**request_params)
            )

            # Check for errors in the response object
            if response.error:
                logger.error(f"OpenAI API error: {response.error.code} - {response.error.message}")
                raise ProviderError(f"OpenAI API error: {response.error.code} - {response.error.message}")

            # Access cached_tokens from prompt_tokens_details as per OpenAI docs
            cached_tokens = 0
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
            
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_tokens=cached_tokens
            )

            # Extract text content - need to search the output list
            content = ""
            if response.output:
                for output_item in response.output:
                    if output_item.type == "message" and output_item.role == "assistant":
                        if output_item.content:
                            for content_item in output_item.content:
                                if content_item.type == "output_text":
                                    content += content_item.text
                                    # Assuming we only care about the first text block if multiple exist
                                    # or concatenate if needed? Sticking to concatenation for now.
            
            if not content:
                 logger.warning("No output text found in response.")
                 # Check if response.output_text convenience attribute exists and has text
                 if hasattr(response, 'output_text') and isinstance(response.output_text, str):
                     content = response.output_text # Use convenience attribute if primary search fails

            logger.debug(f"Generated response with {usage.input_tokens} input, {usage.output_tokens} output, {usage.cached_tokens} cached tokens")

            return LLMResponse(
                content=content,
                model_name=model, # Return the original model name with prefix
                usage=usage,
                latency=0  # Will be set by ResponseSynthesizer
            )
            
        except Exception as e:
            # Catch specific OpenAI errors if known, otherwise general exception
            logger.error(f"Error during generation: {str(e)}")
            # Check if it's already a ProviderError or ConfigurationError
            if isinstance(e, (ProviderError, ConfigurationError)):
                raise
            raise ProviderError(f"OpenAI generation failed: {str(e)}")

    async def _stream_response(self, 
                        messages: List[Dict[str, Any]], 
                        **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream a response from the OpenAI API (async generator)
        
        Args:
            messages: List of message dictionaries (passed as 'input' to API)
            **kwargs: Additional arguments to pass to the API
        
        Yields:
            Chunks of the generated response
        """
        model = kwargs.pop('model', None)
        if not model:
            raise ValueError("Model parameter is required")

        # Map max_tokens to max_output_tokens
        if 'max_tokens' in kwargs and 'max_output_tokens' not in kwargs:
            kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
        # Ensure max_output_tokens has a default value
        if 'max_output_tokens' not in kwargs:
            kwargs['max_output_tokens'] = 1024

        logger.debug(f"Starting streaming response with model: {model}")
        
        full_response = ""
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        stream = None
        in_reasoning_block = False
        in_code_block = False
        web_search_queries = {}  # Store queries by search ID
        web_search_results = {}  # Store results by search ID
        final_citations = []  # Store all citations for final list
        
        # Strip prefix for API call
        api_model_name = model
        if api_model_name.startswith("responses-"):
            api_model_name = api_model_name[len("responses-"):]
            logger.debug(f"Stripped prefix, using API model name: {api_model_name}")

        request_params = {
            "model": api_model_name, # Use stripped name
            "input": messages,
            "stream": True,
            **kwargs
        }

        try:
            stream = self.client.responses.create(**request_params)
            self._current_generation = stream 

            for event in stream:
                # Process events based on type
                if event.type == 'response.reasoning_summary_text.delta':
                    reasoning_delta = event.delta
                    # Open the block if it's the first reasoning delta
                    if not in_reasoning_block:
                        yield "\n\n<think>\n\n"
                        in_reasoning_block = True
                    yield reasoning_delta

                elif event.type == 'response.output_text.delta':
                    # Close the reasoning block *before* the first output text delta
                    if in_reasoning_block:
                        yield "\n\n</think>\n\n"
                        in_reasoning_block = False # Reasoning is definitely over
                    
                    # Close the code block *before* the first output text delta
                    if in_code_block:
                        yield "\n```\n\n"
                        in_code_block = False
                    
                    # Yield the actual output text delta
                    delta = event.delta
                    full_response += delta
                    yield delta
                
                elif event.type == 'response.code_interpreter_call.in_progress':
                    # Close reasoning block if open when code starts
                    # if in_reasoning_block:
                    #     yield "\n\n</think>\n\n"
                    #     in_reasoning_block = False
                    
                    # Start code block
                    if not in_code_block:
                        yield "\n```py\n"
                        in_code_block = True
                
                elif event.type == 'response.code_interpreter_call_code.delta':
                    # Ensure we're in a code block (should already be from in_progress event)
                    if not in_code_block:
                        yield "\n```py\n"
                        in_code_block = True
                    
                    # Yield the code delta
                    code_delta = event.delta
                    yield code_delta
                
                elif event.type == 'response.code_interpreter_call.completed':
                    # Close code block when interpretation is complete
                    if in_code_block:
                        yield "\n```\n\n"
                        in_code_block = False
                
                elif event.type == 'response.output_item.done':
                    # Only process if event.item exists
                    if hasattr(event, 'item') and event.item is not None:
                        # Handle web search completion
                        if getattr(event.item, 'type', None) == 'web_search_call':
                            try:
                                query = event.item.to_dict().get('action', {}).get('query', 'Unknown')
                                if query != 'Unknown':
                                    yield f"\n**Search Query**: {query}\n"
                                yield "✅ Search completed\n"
                            except:
                                yield "✅ Search completed\n"
                        
                        # Check if this is a code interpreter call with outputs
                        elif (hasattr(event.item, 'type') and 
                            event.item.type == 'code_interpreter_call' and
                            hasattr(event.item, 'outputs') and 
                            event.item.outputs):
                            
                            # Collect all logs first to check if we have any content
                            logs_content = []
                            images = []
                            
                            for output in event.item.outputs:
                                # Handle both dict and object formats
                                output_type = getattr(output, 'type', None) or output.get('type') if isinstance(output, dict) else None
                                
                                if output_type == 'logs':
                                    logs = getattr(output, 'logs', None) or output.get('logs') if isinstance(output, dict) else None
                                    if logs and logs.strip():  # Only add non-empty logs
                                        logs_content.append(logs)
                                elif output_type == 'image':
                                    url = getattr(output, 'url', None) or output.get('url') if isinstance(output, dict) else None
                                    if url:
                                        images.append(url)
                            
                            # Only display logs block if we have actual content
                            if logs_content:
                                yield "\n```plaintext\n"
                                for logs in logs_content:
                                    yield logs
                                yield "\n```\n\n"
                            
                            # Display images
                            for url in images:
                                yield f"<code_execution_image>{url}</code_execution_image>\n\n"
                        
                        # Handle annotations (file citations and URL citations)
                        if hasattr(event.item, 'content') and event.item.content:
                            for content_item in event.item.content:
                                if hasattr(content_item, 'annotations') and content_item.annotations:
                                    for annotation in content_item.annotations:
                                        if annotation.type == 'container_file_citation':
                                            yield f"\n\n<code_execution_file><code_execution_file_id>{annotation.file_id}</code_execution_file_id><code_execution_file_name>{annotation.filename}</code_execution_file_name><code_execution_container_id>{annotation.container_id}</code_execution_container_id></code_execution_file>\n\n"
                                        elif annotation.type == 'url_citation' and annotation.url and annotation.title:
                                            citation_entry = {'url': annotation.url, 'title': annotation.title}
                                            if citation_entry not in final_citations:
                                                final_citations.append(citation_entry)
                
                elif event.type == 'response.web_search_call.in_progress':
                    yield "\n🔍 **Starting Web Search**\n"
                
                elif event.type == 'response.completed':
                    # Close reasoning block if it was still open when completion event arrives
                    if in_reasoning_block:
                         yield "\n\n</think>\n\n"
                         in_reasoning_block = False
                    
                    # Close code block if it was still open when completion event arrives
                    if in_code_block:
                         yield "\n```\n\n"
                         in_code_block = False
                    
                    # Process usage etc.
                    if hasattr(event, 'usage'):
                        usage = event.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        
                        # Access cached_tokens from prompt_tokens_details as per OpenAI docs
                        cached_tokens = 0
                        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                            cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                        
                        self.last_usage = Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_tokens=cached_tokens
                        )
                        logger.debug(f"Completed streaming with {input_tokens} input, {output_tokens} output, {cached_tokens} cached tokens")
                    break # Exit loop on completion
                
                elif event.type == 'error':
                    # Close reasoning block if open on error
                    if in_reasoning_block:
                         yield "\n\n</think>\n\n"
                         in_reasoning_block = False
                    
                    # Close code block if open on error
                    if in_code_block:
                         yield "\n```\n\n"
                         in_code_block = False
                    
                    logger.error(f"OpenAI streaming error: {event.code} - {event.message}")
                    raise ProviderError(f"OpenAI streaming error: {event.code} - {event.message}")

                # Note: We are IGNORING other event types like response.reasoning_summary_part.added/done
                # for the purpose of tag placement. We only care about the transition from reasoning deltas
                # to output text deltas.
            
            # Store the full response after successful streaming
            self.last_response = full_response
                    
        except Exception as e:
            # Close reasoning block if an exception occurs during the loop
            if in_reasoning_block:
                 # Attempt to yield closing tag, might fail if connection broken
                 try: yield "\n\n</think>\n\n" 
                 except: pass 
                 in_reasoning_block = False
            
            # Close code block if an exception occurs during the loop
            if in_code_block:
                 # Attempt to yield closing tag, might fail if connection broken
                 try: yield "\n```\n\n" 
                 except: pass 
                 in_code_block = False
            
            logger.error(f"Error during streaming: {str(e)}")
            if isinstance(e, ProviderError):
                 raise
            raise ProviderError(f"OpenAI streaming failed: {str(e)}")
            
        finally:
            # Final check: Ensure reasoning block is closed if stream ends for any reason
            if in_reasoning_block:
                 # Attempt to yield closing tag
                 try: yield "\n\n</think>\n\n" 
                 except: pass 
            
            # Final check: Ensure code block is closed if stream ends for any reason
            if in_code_block:
                 # Attempt to yield closing tag
                 try: yield "\n```\n\n" 
                 except: pass 
            
            # Output final citation list if not already done
            if final_citations:
                try:
                    yield "\n<citation_list>\n"
                    for citation in final_citations:
                        yield f"<citation_source><citation_url>{citation['url']}</citation_url></citation_source>\n"
                    yield "</citation_list>\n"
                except: pass
            
            self._current_generation = None

    def stop_generation(self):
        """Stop the current generation if any"""
        if self._current_generation:
            logger.info("Attempting to stop current OpenAI generation")
            # Similar to Anthropic, just clearing the reference might be enough
            # if iteration stops. If the library offers a specific method, call it here.
            # e.g., if self._current_generation has a close() method:
            # try:
            #     self._current_generation.close()
            # except Exception as e:
            #     logger.warning(f"Could not explicitly close stream: {e}")
            self._current_generation = None
            logger.info("OpenAI generation reference cleared.") 