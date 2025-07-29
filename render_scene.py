import subprocess
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/render-scene")
async def render_scene():
    # Run Puppeteer render script
    result = subprocess.run(["node", "render.js"], capture_output=True)
    if result.returncode != 0:
        return {"error": "Rendering failed", "details": result.stderr.decode()}

    # Serve the rendered image
    return FileResponse("frame.png", media_type="image/png")
