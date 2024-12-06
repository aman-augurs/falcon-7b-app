import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Model and tokenizer
    base_model = "aman-augurs/falcon7b-fine-tuned"

    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load the model with updated configuration
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return tokenizer, model, device

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def generate_response(tokenizer, model, device, prompt):
    try:
        # Encode input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate output with temperature setting
        outputs = model.generate(
            inputs['input_ids'],
            max_length=200,
            num_return_sequences=1,
            temperature=0.1,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "An error occurred during response generation."

def main():
    # Set page configuration
    st.set_page_config(page_title="Falcon 7B Query App", page_icon="🤖")

    # Title and description
    st.title("Falcon 7B Query Interface")
    st.markdown("Ask a query and get AI-generated responses!")

    # Check CUDA availability
    if not torch.cuda.is_available():
        st.warning("CUDA is not available. The app will run on CPU, which may be slower.")

    # Load model (only once)
    if 'tokenizer' not in st.session_state:
        tokenizer, model, device = load_model()
        
        if tokenizer and model:
            st.session_state['tokenizer'] = tokenizer
            st.session_state['model'] = model
            st.session_state['device'] = device
        else:
            st.error("Failed to load the model. Please check your setup.")
            return

    # Input text area for user query
    user_prompt = st.text_area(
        "Enter your query:", 
        placeholder="Type your question here...", 
        height=100
    )

    # Generate button
    if st.button("Generate Response"):
        if user_prompt:
            try:
                # Show loading spinner
                with st.spinner('Generating response...'):
                    response = generate_response(
                        st.session_state['tokenizer'], 
                        st.session_state['model'], 
                        st.session_state['device'], 
                        user_prompt
                    )
                
                # Display the generated response
                st.subheader("AI Response:")
                st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

    # Optional: Add some styling
    st.markdown("""
    <style>
    .stTextArea textarea {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()