"""
Final VRP Streamlit app (presentation-ready)

Features:
- No Excel: user inputs everything on the frontend
- OR-Tools solver (capacity + distance + node-drop penalty)
- Polished route visualization (auto-scaled coordinates, legible labels, legend)
- Auto-generated PDF report (2-3 pages) embedding the route plot and including:
    - Title page with student names and course
    - Model formulation (objective + constraints explained)
    - Solution summary table and key metrics
    - Embedded route map image
- Team: Vaishali Anand, Vipul Yadav, Kundan, Arpit Agrawal
- Course: MSL304 (you can edit header text below if needed)
"""

import io
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# App config & Header
# ----------------------------
st.set_page_config(page_title="VRP Solver — Final (PDF + Polished Viz)", layout="wide")
st.title("🚛 Vehicle Routing Problem — Interactive Solver (Final)")
st.markdown(
    "Team: **Vaishali Anand, Vipul Yadav, Kundan, Arpit Agrawal**  \nCourse: **MSL304**"
)
st.markdown("---")

# ----------------------------
# User Inputs (problem definition)
# ----------------------------
with st.expander("Problem size & basic settings (open to edit)", expanded=True):
    cols = st.columns(3)
    with cols[0]:
        num_locations = st.number_input(
            "Number of locations (including depot)", min_value=2, value=9, step=1
        )
    with cols[1]:
        num_vehicles = st.number_input("Number of vehicles", min_value=1, value=3, step=1)
    with cols[2]:
        depot_index = st.number_input(
            "Depot index (0-based)", min_value=0, value=0, max_value=num_locations - 1, step=1
        )

# default matrix/weights generator functions
@st.cache_data
def default_matrix(n):
    rng = np.random.default_rng(42)
    base = rng.integers(10, 150, size=(n, n))
    mat = (base + base.T) // 2
    np.fill_diagonal(mat, 0)
    df = pd.DataFrame(mat, index=[str(i) for i in range(n)], columns=[str(i) for i in range(n)])
    return df


@st.cache_data
def default_weights(n, low=5, high=50):
    rng = np.random.default_rng(21)
    return pd.Series(rng.integers(low, high, size=n), index=[str(i) for i in range(n)])


# Distance matrix editor
st.subheader("1) Distance Matrix (editable)")
if "distance_df" not in st.session_state or st.session_state.get("dm_n", None) != num_locations:
    st.session_state["distance_df"] = default_matrix(num_locations)
    st.session_state["dm_n"] = num_locations

distance_df = st.experimental_data_editor(st.session_state["distance_df"], num_rows="fixed", key="dist_editor")
try:
    dist_mat = distance_df.astype(int).values
except Exception as e:
    st.error(f"Distance matrix must contain numeric values. {e}")
    st.stop()

# enforce symmetry & zero diagonal
for i in range(len(dist_mat)):
    dist_mat[i, i] = 0
    for j in range(i + 1, len(dist_mat)):
        avg = int((dist_mat[i, j] + dist_mat[j, i]) / 2)
        dist_mat[i, j] = avg
        dist_mat[j, i] = avg

# Weights editor (demand)
st.subheader("2) Demand (weight) per location")
if "weights" not in st.session_state or st.session_state.get("w_n", None) != num_locations:
    st.session_state["weights"] = default_weights(num_locations)
    st.session_state["w_n"] = num_locations

weights_df = st.experimental_data_editor(
    pd.DataFrame({"Weight": st.session_state["weights"]}), num_rows="fixed", key="weights_editor"
)
try:
    weights = [int(x) for x in weights_df["Weight"].values]
except Exception as e:
    st.error(f"Weight column must be numeric. {e}")
    st.stop()

# Vehicle parameters (capacity & max distance)
st.subheader("3) Vehicle parameters")
vehicle_caps = []
vehicle_maxdist = []
vehicle_cols = st.columns(min(4, num_vehicles))
for i in range(num_vehicles):
    with vehicle_cols[i % len(vehicle_cols)]:
        cap = st.number_input(f"Vehicle {i} capacity (weight units)", value=100, min_value=1, key=f"cap_{i}")
        md = st.number_input(f"Vehicle {i} max distance", value=800, min_value=1, key=f"md_{i}")
    vehicle_caps.append(int(cap))
    vehicle_maxdist.append(int(md))

# solver options
st.subheader("4) Solver options")
penalty = st.number_input("Penalty for dropping a node (0 = no drop allowed)", min_value=0, value=10000, step=100)
time_limit = st.slider("Solver time limit (seconds)", min_value=5, max_value=120, value=20)
first_strategy = st.selectbox(
    "First solution strategy",
    options=[
        "PATH_CHEAPEST_ARC",
        "SAVINGS",
        "SWEEP",
        "CHRISTOFIDES",
        "ALL_UNPERFORMED",
        "BEST_INSERTION",
    ],
)
metaheuristic = st.selectbox(
    "Local search metaheuristic",
    options=["GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "SIMULATED_ANNEALING", "AUTOMATIC"],
)

st.markdown("---")

# ----------------------------
# Solve VRP button
# ----------------------------
if st.button("🔎 Solve VRP"):
    N = num_locations

    # Input validation
    if len(dist_mat) != N or any(len(row) != N for row in dist_mat):
        st.error("Distance matrix size does not match number of locations.")
        st.stop()
    if len(weights) != N:
        st.error("Weights vector size does not match number of locations.")
        st.stop()
    if depot_index < 0 or depot_index >= N:
        st.error("Depot index out of range.")
        st.stop()

    # Build OR-Tools model
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_mat[from_node][to_node])

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Capacity (weight) dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(weights[node])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, vehicle_caps, True, "Weight")

    # Distance dimension and per-vehicle limit
    routing.AddDimension(transit_cb_idx, 0, max(vehicle_maxdist), True, "Distance")
    distance_dim = routing.GetDimensionOrDie("Distance")
    for i in range(num_vehicles):
        distance_dim.CumulVar(routing.End(i)).SetMax(vehicle_maxdist[i])

    # Allow dropping nodes with penalty (except depot)
    for node in range(N):
        if node == depot_index:
            continue
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Configure solver search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    strategy_map = {
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "SWEEP": routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        "ALL_UNPERFORMED": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        "BEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
    }
    meta_map = {
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "AUTOMATIC": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    }
    search_params.first_solution_strategy = strategy_map.get(first_strategy, routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.local_search_metaheuristic = meta_map.get(metaheuristic, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_params.time_limit.seconds = int(time_limit)

    with st.spinner("Solving — OR-Tools running..."):
        solution = routing.SolveWithParameters(search_params)

    if not solution:
        st.error("No solution found. Try relaxing constraints, increasing time limit, or raising penalty.")
        st.stop()

    # ----------------------------
    # Extract solution
    # ----------------------------
    routes = []
    total_distance = 0
    total_load = 0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route_nodes = []
        route_load = 0
        route_dist = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            route_load += weights[node]
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                curr_node = manager.IndexToNode(prev_index)
                next_node = manager.IndexToNode(index)
                route_dist += dist_mat[curr_node][next_node]
        route_nodes.append(depot_index)  # end at depot
        if len(route_nodes) > 2:  # vehicle did some deliveries
            routes.append({"Vehicle": v, "Route": route_nodes, "Distance": route_dist, "Load": route_load})
            total_distance += route_dist
            total_load += route_load

    df_routes = pd.DataFrame(routes)

    # ----------------------------
    # Solution display
    # ----------------------------
    st.subheader("Solution Summary")
    if df_routes.empty:
        st.warning("No routes were used (all nodes may have been dropped).")
    else:
        st.dataframe(df_routes.style.format({"Route": lambda r: "→".join(map(str, r))}))
        st.success(f"Total distance: {total_distance} units | Total load assigned: {total_load} units")

    # ----------------------------
    # Polished Visualization
    # ----------------------------
    st.subheader("Route Map (polished & auto-scaled)")
    # create pseudo-geographic coordinates that look natural
    rng = np.random.default_rng(10)  # deterministic placement for reproducibility
    coords = rng.random((N, 2)) * 100

    xs = coords[:, 0]
    ys = coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    # margins: expand axis limits slightly
    xpad = (xs.max() - xs.min()) * 0.08 if xs.max() != xs.min() else 5
    ypad = (ys.max() - ys.min()) * 0.08 if ys.max() != ys.min() else 5
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    cmap = plt.get_cmap("tab10")
    # draw nodes
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == depot_index:
            ax.scatter(x, y, s=220, edgecolor="k", linewidth=1.2, zorder=4, facecolor="gold")
            ax.annotate(f"{i}\nDepot\nW:{weights[i]}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
        else:
            ax.scatter(x, y, s=90, edgecolor="k", linewidth=0.6, zorder=3, facecolor="lightgray")
            ax.annotate(f"{i}\nW:{weights[i]}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # draw routes with arrows and a legend
    for r in routes:
        veh = r["Vehicle"]
        route = r["Route"]
        pts = [(xs[n], ys[n]) for n in route]
        rx, ry = zip(*pts)
        ax.plot(rx, ry, color=cmap(veh % 10), linewidth=2.6, alpha=0.9, zorder=2)
        # draw small arrowheads along the path
        for k in range(len(pts) - 1):
            ax.annotate(
                "",
                xy=pts[k + 1],
                xytext=pts[k],
                arrowprops=dict(arrowstyle="-|>", color=cmap(veh % 10), lw=1.6),
                zorder=2,
            )

    # build legend entries
    handles = []
    labels = []
    for r in routes:
        v = r["Vehicle"]
        labels.append(f"Vehicle {v} (Load:{r['Load']}, Dist:{r['Distance']})")
        handles.append(plt.Line2D([0], [0], color=cmap(v % 10), lw=3))
    if handles:
        ax.legend(handles, labels, title="Routes", loc="upper right", bbox_to_anchor=(1.25, 1.03))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Optimized Vehicle Routes", fontsize=14)
    plt.tight_layout()

    st.pyplot(fig)

    # ----------------------------
    # PDF REPORT GENERATION (2-3 pages)
    # ----------------------------
    st.subheader("Downloadable PDF Report (2-3 pages)")

    def build_pdf_bytes(title_text="VRP Report"):
        """
        Generate a multi-page PDF in-memory and return bytes buffer.
        Pages:
          1) Title page: project, team, course
          2) Model formulation + parameters + results table
          3) Embedded route image + closing notes
        """
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        Story = []

        # Title page
        Story.append(Paragraph("<b>Vehicle Routing Problem — Project Report</b>", styles["Title"]))
        Story.append(Spacer(1, 12))
        Story.append(Paragraph(f"<b>Course:</b> MSL304", styles["Normal"]))
        Story.append(Paragraph(f"<b>Team:</b> Vaishali Anand, Vipul Yadav, Kundan, Arpit Agrawal", styles["Normal"]))
        Story.append(Spacer(1, 12))
        Story.append(Paragraph("<b>Abstract</b>", styles["Heading2"]))
        Story.append(Paragraph("This report describes a capacity-constrained vehicle routing problem (VRP) solved using OR-Tools. The app allows interactive specification of distances, demands, and vehicle constraints; it computes feasible routes minimizing total travel cost while respecting capacity and distance constraints.", styles["Normal"]))
        Story.append(PageBreak())

        # Model formulation page
        Story.append(Paragraph("<b>Model Formulation</b>", styles["Heading2"]))
        # Present equations in plain text (clear and exam-friendly)
        model_text = """
        Objective: Minimize total travel distance (or cost).
        Let x_{i,j}^k = 1 if vehicle k travels from node i to node j, else 0.

        Minimize:  Σ_k Σ_i Σ_j c_{i,j} * x_{i,j}^k

        Subject to:
        1) Each customer is visited at most once: Σ_k Σ_j x_{i,j}^k = 1  (for all customers i)
        2) Flow conservation (route continuity) for each vehicle.
        3) Capacity constraint: For each vehicle k, Σ_i demand_i * visit_i^k ≤ capacity_k.
        4) Maximum route distance per vehicle: distance_k ≤ MaxDistance_k.
        5) Depot start/end: each vehicle route starts and ends at the depot.
        6) Optional node-drop penalty: if infeasible, nodes may be left unserved at penalty cost.
        """
        Story.append(Paragraph(model_text, styles["Normal"]))
        Story.append(Spacer(1, 12))

        # Parameters & results summary
        Story.append(Paragraph("<b>Parameters & Results</b>", styles["Heading2"]))
        param_lines = [
            ["Parameter", "Value"],
            ["Locations (including depot)", str(N)],
            ["Depot index", str(depot_index)],
            ["Vehicles", str(num_vehicles)],
            ["Vehicle capacities", str(vehicle_caps)],
            ["Vehicle max distances", str(vehicle_maxdist)],
            ["Penalty for dropping node", str(penalty)],
            ["Solver time limit (s)", str(time_limit)],
        ]
        param_table = Table(param_lines, hAlign="LEFT", colWidths=[200, 260])
        param_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        Story.append(param_table)
        Story.append(Spacer(1, 12))

        # routes table
        tbl_data = [["Vehicle", "Route", "Distance", "Load"]]
        for _, row in df_routes.iterrows():
            tbl_data.append([str(row["Vehicle"]), " → ".join(map(str, row["Route"])), str(row["Distance"]), str(row["Load"])])

        routes_table = Table(tbl_data, hAlign="LEFT", colWidths=[70, 280, 80, 80])
        routes_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.gray), ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke)]))
        Story.append(routes_table)
        Story.append(Spacer(1, 12))
        Story.append(Paragraph(f"<b>Total Distance:</b> {total_distance}   <b>Total Load assigned:</b> {total_load}", styles["Normal"]))
        Story.append(PageBreak())

        # Embedded route image page
        Story.append(Paragraph("<b>Route Map</b>", styles["Heading2"]))
        Story.append(Spacer(1, 6))

        # Save the matplotlib figure into a PNG in-memory and embed
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
        img_buf.seek(0)
        # Use RLImage to embed PNG
        rl_img = RLImage(img_buf, width=450, height=300)  # scale to fit page
        Story.append(rl_img)
        Story.append(Spacer(1, 12))
        Story.append(Paragraph("Notes: Routes are colored by vehicle. Depot is highlighted. Labels show node index and weight demand.", styles["Normal"]))
        Story.append(Spacer(1, 12))

        # Closing
        Story.append(Paragraph("<b>Conclusions</b>", styles["Heading2"]))
        Story.append(Paragraph("The app solves VRP instances interactively and provides route visualizations and reports. For large instances, increase the solver time limit for better solutions.", styles["Normal"]))
        Story.append(Spacer(1, 12))

        doc.build(Story)
        buf.seek(0)
        return buf

    pdf_buf = build_pdf_bytes()
    st.download_button("📥 Download PDF Report (with embedded route map)", data=pdf_buf.getvalue(), file_name="VRP_Report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Polished VRP app — Team: Vaishali Anand, Vipul Yadav, Kundan, Arpit Agrawal | MSL304")
