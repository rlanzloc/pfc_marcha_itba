from app import app
server = app.server  # ¡Esta línea es clave!

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=5000)
