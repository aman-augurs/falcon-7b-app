import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    # Check if CUDA is available and set the device accordingly
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        st.warning("CUDA not found. Using CPU instead.")

    # Model and tokenizer
    base_model = "aman-augurs/falcon7b-fine-tuned"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    return tokenizer, model, device

def generate_response(tokenizer, model, device, prompt):
    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output with temperature setting
    with torch.no_grad():
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

def main():
    # Set page title and icon
    st.set_page_config(page_title="Claim Query App", page_icon="🤖")

    # Title and description
    st.title("Falcon 7B based Query Interface")
    st.markdown("Ask a query Related to Claim data and get AI-generated responses!")

    # Load model (only once)
    if 'tokenizer' not in st.session_state:
        try:
            tokenizer, model, device = load_model()
            st.session_state['tokenizer'] = tokenizer
            st.session_state['model'] = model
            st.session_state['device'] = device
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    # Input text area for user query
    user_prompt = st.text_area("Enter your query:", 
                                placeholder="Type your question here...", 
                                height=100)

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