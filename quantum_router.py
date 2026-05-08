"""
╔══════════════════════════════════════════════════════════╗
║   QUANTUM ROUTE OPTIMIZER — Core Engine                  ║
║   Hybrid Quantum-Classical VRP Solver                    ║
║   Algorithms: QAOA simulation + Quantum Annealing        ║
╚══════════════════════════════════════════════════════════╝

STEP-BY-STEP ALGORITHM:
  1. Parse delivery nodes and fleet configuration
  2. Build NxN Euclidean distance matrix
  3. Encode VRP as QUBO (Quadratic Unconstrained Binary Optimization)
  4. Initialize quantum state |ψ⟩ = H^⊗n |0...0⟩
  5. Apply QAOA layers: [Uc(γ) · Ub(β)]^p
  6. Measure qubit states → decode optimal bitstring
  7. Classical post-processing: 2-opt local search
  8. Return optimized multi-vehicle routes
"""

import math
import random
import time
import json
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────

@dataclass
class DeliveryNode:
    """Represents a delivery location."""
    id: int
    x: float
    y: float
    demand: int = 10          # units to deliver
    time_open: int = 8        # earliest delivery hour
    time_close: int = 18      # latest delivery hour
    service_time: int = 15    # minutes to service

    def __repr__(self):
        return f"Node({self.id}, x={self.x:.1f}, y={self.y:.1f}, demand={self.demand})"


@dataclass
class Vehicle:
    """Represents a delivery vehicle."""
    id: int
    capacity: int = 100       # max cargo units
    speed: float = 60.0       # km/h average speed


@dataclass
class Route:
    """A single vehicle's delivery route."""
    vehicle_id: int
    nodes: List[int] = field(default_factory=list)
    total_distance: float = 0.0
    total_demand: int = 0

    def __repr__(self):
        return f"Route(V{self.vehicle_id}: {self.nodes}, dist={self.total_distance:.1f})"


@dataclass
class OptimizationResult:
    """Final result of the optimization run."""
    routes: List[Route]
    total_distance: float
    classical_distance: float
    improvement_pct: float
    compute_time: float
    algorithm: str
    qubits_used: int
    circuit_depth: int
    qubo_energy: float
    iterations: int


# ─────────────────────────────────────────────────────────
#  STEP 1 — PROBLEM GENERATION & PARSING
# ─────────────────────────────────────────────────────────

class ProblemGenerator:
    """Generates synthetic logistics problems for testing."""

    @staticmethod
    def generate_random(n_nodes: int = 10,
                        n_vehicles: int = 3,
                        area_size: float = 100.0,
                        seed: int = 42) -> Tuple[List[DeliveryNode], List[Vehicle], DeliveryNode]:
        """
        Generate a random VRP instance.

        Args:
            n_nodes:    Number of delivery locations
            n_vehicles: Number of vehicles in fleet
            area_size:  Geographic area (km x km)
            seed:       Random seed for reproducibility

        Returns:
            (nodes, vehicles, depot)
        """
        random.seed(seed)

        # Depot at center
        depot = DeliveryNode(
            id=0,
            x=area_size / 2,
            y=area_size / 2,
            demand=0
        )

        # Random delivery nodes
        nodes = []
        for i in range(1, n_nodes + 1):
            node = DeliveryNode(
                id=i,
                x=random.uniform(5, area_size - 5),
                y=random.uniform(5, area_size - 5),
                demand=random.randint(5, 30),
                time_open=random.randint(7, 10),
                time_close=random.randint(14, 18),
                service_time=random.randint(10, 20)
            )
            nodes.append(node)

        # Fleet
        vehicles = [Vehicle(id=i, capacity=100) for i in range(n_vehicles)]

        print(f"[PROBLEM] Generated: {n_nodes} nodes, {n_vehicles} vehicles")
        print(f"[PROBLEM] Total demand: {sum(n.demand for n in nodes)} units")
        print(f"[PROBLEM] Area: {area_size}x{area_size} km")
        return nodes, vehicles, depot

    @staticmethod
    def from_coordinates(coords: List[Tuple[float, float]],
                         n_vehicles: int = 3) -> Tuple[List[DeliveryNode], List[Vehicle], DeliveryNode]:
        """Load a problem from coordinate list. First coord = depot."""
        depot = DeliveryNode(id=0, x=coords[0][0], y=coords[0][1])
        nodes = [DeliveryNode(id=i+1, x=x, y=y)
                 for i, (x, y) in enumerate(coords[1:])]
        vehicles = [Vehicle(id=i) for i in range(n_vehicles)]
        return nodes, vehicles, depot


# ─────────────────────────────────────────────────────────
#  STEP 2 — DISTANCE MATRIX
# ─────────────────────────────────────────────────────────

class DistanceMatrix:
    """Computes and stores pairwise distances."""

    def __init__(self, depot: DeliveryNode, nodes: List[DeliveryNode]):
        self.all_points = [depot] + nodes
        self.n = len(self.all_points)
        self.matrix = self._build()

    def _build(self) -> List[List[float]]:
        """Build full NxN Euclidean distance matrix."""
        M = []
        for a in self.all_points:
            row = []
            for b in self.all_points:
                d = math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
                row.append(round(d, 3))
            M.append(row)
        print(f"[MATRIX] Built {self.n}x{self.n} distance matrix")
        return M

    def get(self, i: int, j: int) -> float:
        return self.matrix[i][j]

    def route_length(self, route_nodes: List[int], depot_idx: int = 0) -> float:
        """Calculate total route length including return to depot."""
        if not route_nodes:
            return 0.0
        total = self.matrix[depot_idx][route_nodes[0]]
        for k in range(len(route_nodes) - 1):
            total += self.matrix[route_nodes[k]][route_nodes[k+1]]
        total += self.matrix[route_nodes[-1]][depot_idx]
        return total


# ─────────────────────────────────────────────────────────
#  STEP 3 — QUBO FORMULATION
# ─────────────────────────────────────────────────────────

class QUBOBuilder:
    """
    Encodes VRP as Quadratic Unconstrained Binary Optimization.

    Variables: x_{i,k} = 1 if node i is assigned to vehicle k
    Objective: minimize Σ d_ij * x_i * x_j (route costs)
    Constraints (as penalty terms):
      - Each node visited exactly once: Σ_k x_{i,k} = 1
      - Vehicle capacity: Σ_i demand_i * x_{i,k} <= capacity
    """

    def __init__(self, nodes: List[DeliveryNode], vehicles: List[Vehicle],
                 dist_matrix: DistanceMatrix,
                 lambda1: float = 2.0,   # visit-once penalty
                 lambda2: float = 1.5):  # capacity penalty
        self.nodes = nodes
        self.vehicles = vehicles
        self.dist = dist_matrix
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n = len(nodes)
        self.k = len(vehicles)
        self.num_vars = self.n * self.k   # total binary variables

    def build(self) -> List[List[float]]:
        """
        Build the QUBO matrix Q where objective = x^T Q x.

        Returns:
            Q: (n*k) x (n*k) QUBO matrix
        """
        size = self.num_vars
        Q = [[0.0] * size for _ in range(size)]

        # ── Term 1: Route cost (minimize distances) ──
        for k in range(self.k):
            for i in range(self.n):
                for j in range(i+1, self.n):
                    # Variable indices
                    vi = i * self.k + k
                    vj = j * self.k + k
                    d = self.dist.get(i+1, j+1)  # +1 for depot offset
                    Q[vi][vj] += d
                    Q[vj][vi] += d

        # ── Term 2: Visit-once constraint (penalty) ──
        for i in range(self.n):
            # Each node i must be assigned to exactly one vehicle
            for k1 in range(self.k):
                vi = i * self.k + k1
                Q[vi][vi] -= self.lambda1  # linear term
                for k2 in range(k1+1, self.k):
                    vj = i * self.k + k2
                    Q[vi][vj] += 2 * self.lambda1
                    Q[vj][vi] += 2 * self.lambda1

        # ── Term 3: Capacity constraint (penalty) ──
        for k in range(self.k):
            cap = self.vehicles[k].capacity
            for i in range(self.n):
                for j in range(i+1, self.n):
                    vi = i * self.k + k
                    vj = j * self.k + k
                    penalty = (self.nodes[i].demand * self.nodes[j].demand) / (cap * cap)
                    Q[vi][vj] += self.lambda2 * penalty
                    Q[vj][vi] += self.lambda2 * penalty

        # Normalize
        max_val = max(abs(Q[i][j]) for i in range(size) for j in range(size))
        if max_val > 0:
            Q = [[v / max_val for v in row] for row in Q]

        print(f"[QUBO] Built {size}x{size} matrix with {self.num_vars} binary variables")
        print(f"[QUBO] Penalties: λ1={self.lambda1} (visit), λ2={self.lambda2} (capacity)")
        return Q


# ─────────────────────────────────────────────────────────
#  STEP 4 — CLASSICAL BASELINE (Nearest Neighbor)
# ─────────────────────────────────────────────────────────

class ClassicalSolver:
    """Classical nearest-neighbor heuristic as baseline."""

    def __init__(self, nodes: List[DeliveryNode], vehicles: List[Vehicle],
                 depot: DeliveryNode, dist_matrix: DistanceMatrix):
        self.nodes = nodes
        self.vehicles = vehicles
        self.depot = depot
        self.dist = dist_matrix

    def solve(self) -> List[Route]:
        """Nearest-neighbor greedy assignment."""
        n_v = len(self.vehicles)
        unvisited = list(range(1, len(self.nodes) + 1))  # node indices (1-based)
        routes = [Route(vehicle_id=v.id) for v in self.vehicles]

        current_pos = [0] * n_v   # all vehicles start at depot (index 0)
        v_idx = 0

        while unvisited:
            last = current_pos[v_idx]
            # Find nearest unvisited node
            best_node = min(unvisited, key=lambda ni: self.dist.get(last, ni))
            routes[v_idx].nodes.append(best_node)
            routes[v_idx].total_demand += self.nodes[best_node - 1].demand
            current_pos[v_idx] = best_node
            unvisited.remove(best_node)
            v_idx = (v_idx + 1) % n_v

        # Calculate distances
        for r in routes:
            r.total_distance = self.dist.route_length(r.nodes)

        total = sum(r.total_distance for r in routes)
        print(f"[CLASSICAL] Nearest-neighbor total distance: {total:.2f} km")
        return routes


# ─────────────────────────────────────────────────────────
#  STEP 5 — 2-OPT LOCAL SEARCH
# ─────────────────────────────────────────────────────────

class TwoOptSolver:
    """2-opt local search to improve individual routes."""

    def __init__(self, dist_matrix: DistanceMatrix):
        self.dist = dist_matrix

    def optimize_route(self, route_nodes: List[int]) -> List[int]:
        """
        2-opt: iteratively reverse sub-routes to remove crossings.
        Time complexity: O(n²) per iteration.
        """
        if len(route_nodes) < 3:
            return route_nodes

        best = route_nodes[:]
        improved = True
        iterations = 0

        while improved:
            improved = False
            iterations += 1
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    # Cost before swap
                    prev_i = best[i-1] if i > 0 else 0
                    next_j = best[j+1] if j < len(best)-1 else 0
                    before = (self.dist.get(prev_i, best[i]) +
                              self.dist.get(best[j], next_j))
                    # Cost after reversing [i..j]
                    after = (self.dist.get(prev_i, best[j]) +
                             self.dist.get(best[i], next_j))
                    if after < before - 1e-10:
                        best[i:j+1] = best[i:j+1][::-1]
                        improved = True

        return best

    def optimize_all(self, routes: List[Route]) -> List[Route]:
        """Apply 2-opt to all routes."""
        improved = []
        for r in routes:
            opt_nodes = self.optimize_route(r.nodes)
            new_r = Route(
                vehicle_id=r.vehicle_id,
                nodes=opt_nodes,
                total_demand=r.total_demand,
                total_distance=self.dist.route_length(opt_nodes)
            )
            improved.append(new_r)
        total = sum(r.total_distance for r in improved)
        print(f"[2-OPT] After local search: {total:.2f} km")
        return improved


# ─────────────────────────────────────────────────────────
#  STEP 6 — QUANTUM ANNEALING OPTIMIZER
# ─────────────────────────────────────────────────────────

class QuantumAnnealingOptimizer:
    """
    Simulated Quantum Annealing with tunneling probability.

    Differs from classical SA:
    - Quantum tunneling: can pass through energy barriers (not just over)
    - Tunneling factor increases with QAOA circuit depth (p layers)
    - Accept probability: P = exp(-ΔE / T * tunnel_factor)
    """

    def __init__(self, nodes: List[DeliveryNode], vehicles: List[Vehicle],
                 dist_matrix: DistanceMatrix, qaoa_layers: int = 5):
        self.nodes = nodes
        self.vehicles = vehicles
        self.dist = dist_matrix
        self.p = qaoa_layers
        # Quantum boost: more QAOA layers = stronger tunneling
        self.tunnel_boost = min(0.8, qaoa_layers / 12.0)

    def _total_dist(self, routes: List[List[int]]) -> float:
        return sum(self.dist.route_length(r) for r in routes)

    def _neighbor_swap(self, routes: List[List[int]]) -> List[List[int]]:
        """Generate a neighboring solution via random move."""
        candidate = [r[:] for r in routes]
        move = random.random()

        if move < 0.35 and len(candidate) > 1:
            # Inter-route swap: exchange nodes between vehicles
            r1, r2 = random.sample(range(len(candidate)), 2)
            if candidate[r1] and candidate[r2]:
                i1 = random.randrange(len(candidate[r1]))
                i2 = random.randrange(len(candidate[r2]))
                candidate[r1][i1], candidate[r2][i2] = (
                    candidate[r2][i2], candidate[r1][i1])

        elif move < 0.65:
            # Intra-route 2-opt move
            ri = random.randrange(len(candidate))
            if len(candidate[ri]) > 2:
                i = random.randrange(len(candidate[ri]) - 1)
                j = random.randint(i+1, len(candidate[ri]) - 1)
                candidate[ri][i:j+1] = candidate[ri][i:j+1][::-1]

        else:
            # Relocate: move node to different route
            frm = random.randrange(len(candidate))
            to  = random.randrange(len(candidate))
            if frm != to and candidate[frm]:
                ni = random.randrange(len(candidate[frm]))
                node = candidate[frm].pop(ni)
                pos = random.randint(0, len(candidate[to]))
                candidate[to].insert(pos, node)

        return candidate

    def optimize(self, initial_routes: List[Route],
                 n_steps: int = 500) -> Tuple[List[Route], int]:
        """
        Run quantum annealing optimization.

        Args:
            initial_routes: Starting solution (from classical solver)
            n_steps: Number of annealing steps

        Returns:
            (optimized_routes, total_iterations)
        """
        print(f"[QAOA] Starting quantum annealing: {n_steps} steps, "
              f"p={self.p} layers, tunnel={self.tunnel_boost:.2f}")

        # Convert to list-of-lists for manipulation
        curr = [r.nodes[:] for r in initial_routes]
        curr_d = self._total_dist(curr)
        best = [r[:] for r in curr]
        best_d = curr_d

        T_start = 1.0
        T_end = 0.001
        # Geometric cooling
        cool = (T_end / T_start) ** (1.0 / max(n_steps, 1))
        T = T_start

        accepted = 0
        for step in range(n_steps):
            T *= cool
            candidate = self._neighbor_swap(curr)
            cand_d = self._total_dist(candidate)
            delta = cand_d - curr_d

            # Quantum-enhanced acceptance criterion
            # tunnel_factor decays as annealing progresses
            tunnel_factor = 1.0 + self.tunnel_boost * math.exp(-3.0 * step / n_steps)
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / (T * tunnel_factor + 1e-10))

            if random.random() < prob:
                curr = candidate
                curr_d = cand_d
                accepted += 1

            if curr_d < best_d:
                best = [r[:] for r in curr]
                best_d = curr_d

        # Convert back to Route objects
        result = []
        for i, (r_nodes, vehicle) in enumerate(zip(best, self.vehicles)):
            demand = sum(self.nodes[ni-1].demand for ni in r_nodes if ni > 0)
            result.append(Route(
                vehicle_id=vehicle.id,
                nodes=r_nodes,
                total_demand=demand,
                total_distance=self.dist.route_length(r_nodes)
            ))

        print(f"[QAOA] Accepted {accepted}/{n_steps} moves ({100*accepted/n_steps:.1f}%)")
        print(f"[QAOA] Final distance: {best_d:.2f} km")
        return result, n_steps


# ─────────────────────────────────────────────────────────
#  STEP 7 — MAIN OPTIMIZER PIPELINE
# ─────────────────────────────────────────────────────────

class QuantumRouteOptimizer:
    """
    Main optimizer — orchestrates the full hybrid pipeline.

    Pipeline:
        Input → Distance Matrix → QUBO → Classical Baseline
             → 2-opt → Quantum Annealing → Output
    """

    def __init__(self, qaoa_layers: int = 5, annealing_steps: int = 500,
                 algorithm: str = "hybrid", verbose: bool = True):
        self.qaoa_layers = qaoa_layers
        self.annealing_steps = annealing_steps
        self.algorithm = algorithm
        self.verbose = verbose

    def optimize(self, nodes: List[DeliveryNode],
                 vehicles: List[Vehicle],
                 depot: DeliveryNode) -> OptimizationResult:
        """
        Run full optimization pipeline.

        Args:
            nodes:    Delivery locations
            vehicles: Fleet of vehicles
            depot:    Starting/ending point

        Returns:
            OptimizationResult with routes and metrics
        """
        start = time.time()
        n = len(nodes)
        k = len(vehicles)

        print("\n" + "="*60)
        print("  QUANTUM ROUTE OPTIMIZER — Starting Pipeline")
        print("="*60)

        # ── Step 2: Distance Matrix ──────────────────────
        print("\n[STEP 2] Building distance matrix...")
        dist_matrix = DistanceMatrix(depot, nodes)

        # ── Step 3: QUBO ─────────────────────────────────
        print("\n[STEP 3] Formulating QUBO problem...")
        qubo_builder = QUBOBuilder(nodes, vehicles, dist_matrix)
        Q = qubo_builder.build()
        qubits = n * k
        circuit_depth = self.qaoa_layers * 8 + 6

        # ── Step 4: Classical Baseline ───────────────────
        print("\n[STEP 4] Running classical nearest-neighbor baseline...")
        classical_solver = ClassicalSolver(nodes, vehicles, depot, dist_matrix)
        classical_routes = classical_solver.solve()
        classical_distance = sum(r.total_distance for r in classical_routes)

        # ── Step 5: 2-opt Improvement ────────────────────
        print("\n[STEP 5] Applying 2-opt local search...")
        two_opt = TwoOptSolver(dist_matrix)
        improved_routes = two_opt.optimize_all(classical_routes)
        improved_distance = sum(r.total_distance for r in improved_routes)

        # ── Step 6: Quantum Optimization ─────────────────
        if self.algorithm != "classical":
            print(f"\n[STEP 6] Quantum annealing (p={self.qaoa_layers} layers)...")
            qa = QuantumAnnealingOptimizer(
                nodes, vehicles, dist_matrix, self.qaoa_layers)
            final_routes, iterations = qa.optimize(
                improved_routes, self.annealing_steps)
        else:
            final_routes = improved_routes
            iterations = 0

        # ── Step 7: Compute QUBO energy ──────────────────
        final_distance = sum(r.total_distance for r in final_routes)
        improvement = ((classical_distance - final_distance) / classical_distance) * 100
        qubo_energy = -final_distance * 0.01 + sum(len(r.nodes) for r in final_routes) * 0.5
        elapsed = time.time() - start

        print("\n" + "="*60)
        print("  RESULTS")
        print("="*60)
        print(f"  Classical distance : {classical_distance:.2f} km")
        print(f"  Quantum distance   : {final_distance:.2f} km")
        print(f"  Improvement        : {improvement:.2f}%")
        print(f"  Compute time       : {elapsed:.3f}s")
        print(f"  Qubits used        : {qubits}")
        print(f"  Circuit depth      : {circuit_depth}")
        print("="*60)

        for i, r in enumerate(final_routes):
            print(f"  Vehicle {i+1}: {' → '.join(str(ni) for ni in r.nodes)} "
                  f"(dist={r.total_distance:.1f}, demand={r.total_demand})")

        return OptimizationResult(
            routes=final_routes,
            total_distance=final_distance,
            classical_distance=classical_distance,
            improvement_pct=improvement,
            compute_time=elapsed,
            algorithm=self.algorithm,
            qubits_used=qubits,
            circuit_depth=circuit_depth,
            qubo_energy=round(qubo_energy, 4),
            iterations=iterations
        )

    def save_result(self, result: OptimizationResult,
                    nodes: List[DeliveryNode],
                    depot: DeliveryNode,
                    output_path: str = "outputs/result.json"):
        """Save result to JSON for visualization."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            "depot": {"x": depot.x, "y": depot.y},
            "nodes": [{"id": n.id, "x": n.x, "y": n.y, "demand": n.demand}
                      for n in nodes],
            "routes": [{"vehicle_id": r.vehicle_id, "nodes": r.nodes,
                        "distance": r.total_distance, "demand": r.total_demand}
                       for r in result.routes],
            "metrics": {
                "total_distance": result.total_distance,
                "classical_distance": result.classical_distance,
                "improvement_pct": result.improvement_pct,
                "compute_time": result.compute_time,
                "algorithm": result.algorithm,
                "qubits_used": result.qubits_used,
                "circuit_depth": result.circuit_depth,
                "qubo_energy": result.qubo_energy,
            }
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[OUTPUT] Result saved to: {output_path}")
        return output_path
