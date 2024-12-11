import streamlit as st
import time
import sys
sys.path.append('../')
from main.cww_app import cww_app


#%%
def main():

    st.title("The College Wealth Wizard")

    user_query = st.text_input("Enter your query:")



    if user_query:
        # Add a progress bar
        with st.spinner("Processing your query..."):
            progress_bar = st.progress(0)  # Initialize progress bar

            # Simulate progress (e.g., update it as your app runs)
            for percent_complete in range(100):
                time.sleep(0.01)  # Simulate some work being done
                progress_bar.progress(percent_complete + 1)

            response = cww_app(user_query)
        st.write(response['generation'])



if __name__ == "__main__":

    main()