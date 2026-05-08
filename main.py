import sys
import io
# Fix Windows console Unicode (lambda λ chars in quantum_router.py logs)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple
import os
import webbrowser
import threading
import time

# Add the current directory to sys.path to import from src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.quantum_router import (
    ProblemGenerator,
    QuantumRouteOptimizer,
)

app = FastAPI(title="Quantum Route Optimization")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
class OptimizeRequest(BaseModel):
    n_nodes: int = 12
    n_vehicles: int = 3
    qaoa_layers: int = 5
    annealing_steps: int = 300
    algorithm: str = "hybrid"
    area_size: float = 100.0
    seed: Optional[int] = 42

class CustomOptimizeRequest(BaseModel):
    coords: List[Tuple[float, float]]
    n_vehicles: int = 3
    qaoa_layers: int = 5
    annealing_steps: int = 300
    algorithm: str = "hybrid"

@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    try:
        nodes, vehicles, depot = ProblemGenerator.generate_random(
            n_nodes=req.n_nodes,
            n_vehicles=req.n_vehicles,
            area_size=req.area_size,
            seed=req.seed,
        )
        optimizer = QuantumRouteOptimizer(
            qaoa_layers=req.qaoa_layers,
            annealing_steps=req.annealing_steps,
            algorithm=req.algorithm,
            verbose=False,
        )
        result = optimizer.optimize(nodes, vehicles, depot)
        return {
            "summary": {
                "algorithm": result.algorithm,
                "n_nodes": req.n_nodes,
                "n_vehicles": req.n_vehicles,
                "classical_dist": round(result.classical_distance, 2),
                "quantum_dist": round(result.total_distance, 2),
                "improvement_pct": round(result.improvement_pct, 2),
                "compute_time": round(result.compute_time, 4),
                "qubits_used": result.qubits_used,
                "circuit_depth": result.circuit_depth,
                "qubo_energy": result.qubo_energy
            },
            "routes": [
                {
                    "vehicle_id": r.vehicle_id,
                    "nodes": r.nodes,
                    "distance": round(r.total_distance, 2),
                    "demand": r.total_demand
                } for r in result.routes
            ],
            "nodes": [{"id": n.id, "x": n.x, "y": n.y, "demand": n.demand} for n in nodes],
            "depot": {"x": depot.x, "y": depot.y}
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from React build
DIST_PATH = os.path.join(BASE_DIR, "frontend", "dist")

if os.path.exists(DIST_PATH):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_PATH, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(DIST_PATH, full_path)
        if full_path != "" and os.path.exists(file_path):
            return FileResponse(file_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        return FileResponse(os.path.join(DIST_PATH, "index.html"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
else:
    @app.get("/")
    async def root():
        return {"message": "Quantum Route API v3.0 is running. Frontend build not found at " + DIST_PATH}

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    import uvicorn
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
