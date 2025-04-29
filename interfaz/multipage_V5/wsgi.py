from app import app

server = app.server  # Esto expone el servidor Flask subyacente de Dash

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run_server(host="0.0.0.0", port=port, debug=False)
