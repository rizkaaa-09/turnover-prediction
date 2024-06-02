import streamlit as st

nama_shelter = ['rizka']

nama = st.text_input('Masukkan nama anda')

if st.button('Submit'):
    have_it = nama.lower() in nama_shelter
    'Anda suhu' if have_it else 'Anda bukan suhu'