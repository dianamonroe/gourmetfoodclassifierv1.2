import streamlit as st

# Título de la aplicación
st.title("Pan Classification App (Vacía)")

# Función principal para arrastrar y soltar imágenes
uploaded_files = st.file_uploader(
    "Sube tus imágenes para clasificar",
    type=["jpg", "png"],
    accept_multiple_files=True
)

# Swipe left and right (dummy logic)
if uploaded_files:
    st.write("Arrastra y suelta las imágenes:")
    for file in uploaded_files:
        st.image(file, use_column_width=True)
        st.write("Clasificación:")
        left, right = st.columns(2)
        with left:
            st.button("Es pan de masa madre", key=f"left-{file.name}")
        with right:
            st.button("No es pan de masa madre", key=f"right-{file.name}")
else:
    st.write("No se han subido imágenes.")

st.write("Esta es una versión básica de la app.")



# your code here
