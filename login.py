import streamlit as st

def login():
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if username == 'myusername' and password == 'mypassword':
            st.success('Logged in as {}'.format(username))
            return True
        else:
            st.error('Invalid username or password')
            return False
