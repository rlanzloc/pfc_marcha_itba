import os 
from app import app

server = app.server  # Esto expone el servidor Flask subyacente de Dash

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"App running on http://0.0.0.0:{port}")
    app.run_server(host="127.0.0.1", port=port, debug=False)
