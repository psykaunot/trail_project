#!/usr/bin/env python3
"""
Semantic Forest Viewer
======================
A standalone script to open and visualize the semantic forest saved by the robot shelver.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import webbrowser
from pathlib import Path

def create_interactive_html(G, output_path='semantic_forest_visualization.html'):
    """Creates an interactive HTML visualization using Pyvis."""
    try:
        from pyvis.network import Network
        
        # Create a network
        net = Network(height="800px", width="100%", notebook=False, directed=True)
        
        # Add nodes with proper attributes
        for node_id, node_data in G.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            label = node_data.get('description', node_id)
            
            # Determine color based on node type
            if node_type == 'book_instance':
                color = '#66c2a5'  # Green for books
            elif node_type == 'cluster':
                color = '#fc8d62'  # Orange for clusters
            else:
                color = '#8da0cb'  # Blue for other nodes
                
            # Add position if available (scaled for visualization)
            position = None
            if 'attributes' in node_data and 'world_position' in node_data['attributes']:
                position = node_data['attributes']['world_position']
                
            # Add node with attributes
            net.add_node(
                node_id, 
                label=label[:20] + '...' if len(label) > 20 else label,
                title=f"Type: {node_type}<br>ID: {node_id}<br>Description: {label}",
                color=color
            )
        
        # Add edges with proper attributes
        for source, target, edge_data in G.edges(data=True):
            relation_type = edge_data.get('type', 'unknown')
            
            # Determine edge color based on relation type
            if relation_type == 'parent-child':
                color = '#386cb0'  # Dark blue for hierarchical
                dashes = False
            elif relation_type in ['above', 'below', 'left', 'right', 'near']:
                color = '#f0027f'  # Pink for spatial
                dashes = True
            else:
                color = '#666666'  # Gray for other relations
                dashes = False
                
            # Add the edge
            net.add_edge(
                source, 
                target, 
                title=relation_type,
                color=color,
                dashes=dashes
            )
        
        # Set physics and interaction options
        net.barnes_hut(gravity=-2000, central_gravity=0.1, spring_length=150)
        net.show_buttons(filter_=['physics'])
        
        # Save the network
        net.save_graph(output_path)
        return True
    except ImportError:
        print("Pyvis not installed. Install with: pip install pyvis")
        return False
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        return False

def create_3d_visualization(G, title="Semantic Forest Visualization"):
    """Creates a 3D matplotlib visualization of the semantic forest with enhanced duplicate highlighting."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = {}
    labels = {}
    colors = []
    node_types = {}
    sizes = []
    markers = []
    observation_counts = {}  # Track observation counts for sizing
    position_history_counts = {}  # Track position history for highlighting potential duplicates
    node_status = {}  # Track picked/placed status
    
    # First, find the robot node
    robot_node = None
    robot_position = None
    
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get('node_type', 'unknown')
        if node_type == 'robot':
            robot_node = node_data
            if 'attributes' in node_data and 'world_position' in node_data['attributes']:
                robot_position = node_data['attributes']['world_position']
                print(f"Found robot position: {robot_position}")
                break
    
    # If we didn't find a robot node in the graph, check if there's a position manually saved
    if robot_position is None:
        try:
            from tools import visualize_semantic_forest
            if hasattr(visualize_semantic_forest, 'robot_position'):
                robot_position = visualize_semantic_forest.robot_position
                print(f"Using robot position from tools module: {robot_position}")
        except Exception as e:
            print(f"Could not get robot position from tools: {e}")
    
    # Process all nodes with enhanced attributes for visualization
    for node_id, node_data in G.nodes(data=True):
        pos = None
        if 'attributes' in node_data and 'world_position' in node_data['attributes']:
            pos = node_data['attributes']['world_position']
        else:
            # Assign random position if no position is available
            pos = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
            
        positions[node_id] = pos
        labels[node_id] = node_data.get('description', node_id)[:20]
        node_type = node_data.get('node_type', 'unknown')
        node_types[node_id] = node_type
        
        # Get observation count for size scaling
        obs_count = 1
        if 'attributes' in node_data:
            attrs = node_data['attributes']
            if 'observation_count' in attrs:
                obs_count = attrs['observation_count']
            elif 'observation_history' in attrs:
                obs_count = len(attrs['observation_history'])
            
            # Track position history count
            if 'position_history' in attrs:
                position_history_counts[node_id] = len(attrs['position_history'])
            
            # Track book status (picked/placed)
            if 'status' in attrs:
                node_status[node_id] = attrs['status']
        
        # Store observation count
        observation_counts[node_id] = obs_count
        
        # Determine color, size and marker based on node type and attributes
        if node_type == 'book_instance':
            # Size based on observation count (more observations = larger marker)
            node_size = 80 + (obs_count * 20)  # Base size + scaling
            
            # Color based on status
            if node_id in node_status:
                if node_status[node_id] == 'picked':
                    colors.append('purple')  # Purple for picked books
                elif node_status[node_id] == 'placed':
                    colors.append('blue')  # Blue for placed books
                else:
                    colors.append('green')  # Green for regular books
            else:
                colors.append('green')  # Default green for regular books
                
            sizes.append(node_size)
            markers.append('o')  # Circle marker for books
        elif node_type == 'cluster':
            colors.append('orange')
            sizes.append(150)
            markers.append('s')  # Square marker for clusters
        elif node_type == 'robot':
            colors.append('red')
            sizes.append(300)
            markers.append('^')  # Triangle marker for robot
        else:
            colors.append('blue')
            sizes.append(80)
            markers.append('o')
    
    # Add robot position if we have one but it's not in the graph
    if robot_position is not None and not any(node_types.get(node_id) == 'robot' for node_id in G.nodes()):
        robot_id = "robot_position"
        positions[robot_id] = robot_position
        labels[robot_id] = "ROBOT"
        colors.append('red')
        sizes.append(300)
        markers.append('^')
        print(f"Added robot position manually: {robot_position}")
    
    # Extract node positions into separate lists for each type
    # First, handle books by their status (regular, picked, placed)
    book_nodes = [node_id for node_id, t in node_types.items() if t == 'book_instance']
    regular_books = []
    picked_books = []
    placed_books = []
    
    for node_id in book_nodes:
        if node_id in node_status:
            if node_status[node_id] == 'picked':
                picked_books.append(node_id)
            elif node_status[node_id] == 'placed':
                placed_books.append(node_id)
            else:
                regular_books.append(node_id)
        else:
            regular_books.append(node_id)
    
    # Plot regular books
    if regular_books:
        valid_books = [node_id for node_id in regular_books if node_id in positions]
        if valid_books:  # Make sure we have books with valid positions
            xs = [positions[node_id][0] for node_id in valid_books]
            ys = [positions[node_id][1] for node_id in valid_books]
            zs = [positions[node_id][2] for node_id in valid_books]
            ss = [observation_counts.get(node_id, 1) * 20 + 80 for node_id in valid_books]  # Size based on observations
            
            # Default green for regular books
            ax.scatter(xs, ys, zs, c='green', s=ss, alpha=0.7, label='Books', marker='o')
            
            # Draw position history rings for books with multiple positions
            for i, node_id in enumerate(valid_books):
                if node_id in position_history_counts and position_history_counts[node_id] > 1:
                    # Draw a highlight ring around books with position history
                    x, y, z = positions[node_id]
                    ax.scatter([x], [y], [z], c='none', s=ss[i]*1.5, 
                               alpha=0.7, edgecolors='yellow', linewidths=2, marker='o')
    
    # Plot picked books
    if picked_books:
        valid_picked = [node_id for node_id in picked_books if node_id in positions]
        if valid_picked:  # Make sure we have books with valid positions
            xs = [positions[node_id][0] for node_id in valid_picked]
            ys = [positions[node_id][1] for node_id in valid_picked]
            zs = [positions[node_id][2] for node_id in valid_picked]
            ss = [observation_counts.get(node_id, 1) * 20 + 80 for node_id in valid_picked]  # Size based on observations
            
            ax.scatter(xs, ys, zs, c='purple', s=ss, alpha=0.8, label='Picked Books', marker='o')
    
    # Plot placed books
    if placed_books:
        valid_placed = [node_id for node_id in placed_books if node_id in positions]
        if valid_placed:  # Make sure we have books with valid positions
            xs = [positions[node_id][0] for node_id in valid_placed]
            ys = [positions[node_id][1] for node_id in valid_placed]
            zs = [positions[node_id][2] for node_id in valid_placed]
            ss = [observation_counts.get(node_id, 1) * 20 + 80 for node_id in valid_placed]  # Size based on observations
            
            ax.scatter(xs, ys, zs, c='blue', s=ss, alpha=0.8, label='Placed Books', marker='o')
    
    # Handle other node types
    for node_type in ['cluster', 'robot', 'unknown']:
        nodes_of_type = [node_id for node_id, t in node_types.items() if t == node_type]
        if not nodes_of_type and node_type == 'robot' and robot_position is not None:
            # Add robot position manually if we have one
            nodes_of_type = ['robot_position']
        
        if nodes_of_type:
            xs = [positions[node_id][0] for node_id in nodes_of_type if node_id in positions]
            ys = [positions[node_id][1] for node_id in nodes_of_type if node_id in positions]
            zs = [positions[node_id][2] for node_id in nodes_of_type if node_id in positions]
            
            if node_type == 'cluster':
                ax.scatter(xs, ys, zs, c='orange', s=150, alpha=0.7, label='Clusters', marker='s')
            elif node_type == 'robot':
                ax.scatter(xs, ys, zs, c='red', s=300, alpha=1.0, label='Robot', marker='^')
                
                # Add vertical line from robot to ground
                for x, y, z in zip(xs, ys, zs):
                    ax.plot([x, x], [y, y], [z, 0], 'r--', linewidth=2)
                    
                    # Add "ROBOT" text label
                    ax.text(x, y, z + 0.3, "ROBOT", color='red', fontweight='bold', fontsize=14)
            else:
                ax.scatter(xs, ys, zs, c='blue', s=80, alpha=0.7, label='Other', marker='o')
    
    # Add node labels (except for robot which we already labeled)
    for node_id, pos in positions.items():
        if node_types.get(node_id) != 'robot' and node_id != 'robot_position':
            ax.text(pos[0], pos[1], pos[2], labels[node_id], fontsize=8)
    
    # Plot edges
    for source, target, edge_data in G.edges(data=True):
        if source in positions and target in positions:
            relation_type = edge_data.get('type', 'unknown')
            
            # Determine edge color based on relation type
            if relation_type == 'parent-child':
                color = 'blue'
                linewidth = 2
            elif relation_type in ['above', 'below', 'left', 'right', 'near']:
                color = 'red'
                linewidth = 1
            else:
                color = 'gray'
                linewidth = 1
            
            ax.plot(
                [positions[source][0], positions[target][0]],
                [positions[source][1], positions[target][1]],
                [positions[source][2], positions[target][2]],
                color=color, linewidth=linewidth, alpha=0.5
            )
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(title)
    
    # Improve layout
    plt.tight_layout()
    return fig

def convert_forest_to_networkx(forest_data):
    """Converts the semantic forest data to a NetworkX graph."""
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in forest_data.get('nodes', {}).items():
        G.add_node(node_id, **node_data)
    
    # Add clusters if they exist
    for cluster_id, cluster_data in forest_data.get('clusters', {}).items():
        G.add_node(cluster_id, **cluster_data)
    
    # Add parent-child relationships
    for node_id, node_data in forest_data.get('nodes', {}).items():
        # Add edges to children
        if 'children' in node_data:
            for child_id in node_data['children']:
                G.add_edge(node_id, child_id, type='parent-child')
        
        # Add edge to parent
        if 'parent' in node_data and node_data['parent']:
            G.add_edge(node_data['parent'], node_id, type='parent-child')
    
    # Add spatial relationships if available
    for node_id, node_data in forest_data.get('nodes', {}).items():
        if 'attributes' in node_data and 'spatial_relations' in node_data['attributes']:
            for relation in node_data['attributes']['spatial_relations']:
                if 'related_to' in relation and 'relation_type' in relation:
                    G.add_edge(
                        node_id, 
                        relation['related_to'], 
                        type=relation['relation_type'],
                        confidence=relation.get('confidence', 1.0)
                    )
    
    return G

def display_forest_details(forest_data):
    """Displays detailed information about the forest in text format with enhanced duplicate detection."""
    nodes = forest_data.get('nodes', {})
    clusters = forest_data.get('clusters', {})
    
    print("=" * 60)
    print(f"SEMANTIC FOREST SUMMARY")
    print("=" * 60)
    print(f"Total nodes: {len(nodes)}")
    print(f"Total clusters: {len(clusters)}")
    
    # Analyze for potential duplicates
    book_positions = {}
    for node_id, node_data in nodes.items():
        if node_data.get('node_type') == 'book_instance' and 'attributes' in node_data:
            attrs = node_data['attributes']
            if 'world_position' in attrs:
                pos = tuple(attrs['world_position'])
                if pos not in book_positions:
                    book_positions[pos] = []
                book_positions[pos].append(node_id)
    
    # Count books with multiple observations
    multi_obs_books = 0
    for node_id, node_data in nodes.items():
        if node_data.get('node_type') == 'book_instance' and 'attributes' in node_data:
            attrs = node_data['attributes']
            if 'observation_count' in attrs and attrs['observation_count'] > 1:
                multi_obs_books += 1
                
    # Count books with position history
    books_with_history = 0
    for node_id, node_data in nodes.items():
        if node_data.get('node_type') == 'book_instance' and 'attributes' in node_data:
            attrs = node_data['attributes']
            if 'position_history' in attrs and len(attrs['position_history']) > 1:
                books_with_history += 1
    
    # Count picked/placed books
    picked_books = 0
    placed_books = 0
    for node_id, node_data in nodes.items():
        if node_data.get('node_type') == 'book_instance' and 'attributes' in node_data:
            attrs = node_data['attributes']
            if 'status' in attrs:
                if attrs['status'] == 'picked':
                    picked_books += 1
                elif attrs['status'] == 'placed':
                    placed_books += 1
    
    # Print statistics
    print(f"Books with multiple observations: {multi_obs_books}")
    print(f"Books with position history: {books_with_history}")
    print(f"Picked books: {picked_books}")
    print(f"Placed books: {placed_books}")
    
    # Print potential duplicates
    potential_dupes = 0
    close_books = []
    for pos, node_ids in book_positions.items():
        # Check for books that are within 0.5m of each other
        for pos2, node_ids2 in book_positions.items():
            if pos == pos2 or tuple(pos) in close_books or tuple(pos2) in close_books:
                continue
                
            # Calculate distance
            try:
                distance = np.linalg.norm(np.array(pos) - np.array(pos2))
                if distance < 0.5:  # 50cm threshold
                    potential_dupes += 1
                    close_books.append(tuple(pos))
                    close_books.append(tuple(pos2))
                    print(f"\nPotential duplicate books found at distance {distance:.3f}m:")
                    
                    # Print details of both books
                    for i, (p, ids) in enumerate([(pos, node_ids), (pos2, node_ids2)]):
                        for nid in ids:
                            node = nodes[nid]
                            desc = node.get('description', 'No description')[:50]
                            print(f"  Book {i+1}: {desc}...")
                            print(f"    ID: {nid}")
                            print(f"    Position: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
                            if 'attributes' in node and 'confidence' in node['attributes']:
                                print(f"    Confidence: {node['attributes']['confidence']:.2f}")
            except Exception as e:
                pass  # Skip if distance calculation fails
    
    # Book instances information
    book_instances = [n for n in nodes.values() if n.get('node_type') == 'book_instance']
    print(f"\n{len(book_instances)} BOOK INSTANCES:")
    print("-" * 60)
    
    # Sort books by status (picked, placed, normal)
    sorted_books = sorted(book_instances, 
                         key=lambda x: (x.get('attributes', {}).get('status', 'z') != 'picked',  # Picked first
                                      x.get('attributes', {}).get('status', 'z') != 'placed'))  # Then placed
    
    for i, book in enumerate(sorted_books, 1):
        description = book.get('description', 'No description')
        position = "Unknown"
        confidence = "Unknown"
        timestamp = "Unknown"
        status = "Unknown"
        position_history = None
        obs_count = 1
        
        if 'attributes' in book:
            attrs = book['attributes']
            if 'world_position' in attrs:
                position = f"({attrs['world_position'][0]:.2f}, {attrs['world_position'][1]:.2f}, {attrs['world_position'][2]:.2f})"
            if 'confidence' in attrs:
                confidence = f"{attrs['confidence']:.2f}"
            if 'timestamp' in attrs:
                timestamp = attrs['timestamp']
            if 'status' in attrs:
                status = attrs['status']
            if 'position_history' in attrs:
                position_history = attrs['position_history']
            if 'observation_count' in attrs:
                obs_count = attrs['observation_count']
            elif 'observation_history' in attrs:
                obs_count = len(attrs['observation_history'])
        
        # Format status for display
        status_display = f"[{status.upper()}]" if status != "Unknown" else ""
        
        # Highlight books with potential issues
        highlight = ""
        if obs_count > 3:
            highlight = "[MULTI-OBS]"
        if position_history and len(position_history) > 3:
            highlight = "[MULTI-POS]"
            
        print(f"{i}. {status_display} {highlight} {description}")
        print(f"   ID: {book.get('node_id', 'Unknown')}")
        print(f"   Position: {position}")
        print(f"   Confidence: {confidence}")
        print(f"   Observations: {obs_count}")
        print(f"   Timestamp: {timestamp}")
        
        # Print position history summary if available
        if position_history and len(position_history) > 1:
            print(f"   Position History: {len(position_history)} positions")
            
            # Calculate total distance moved
            total_distance = 0
            for i in range(1, len(position_history)):
                if 'distance_from_previous' in position_history[i]:
                    total_distance += position_history[i]['distance_from_previous']
            
            print(f"   Total distance moved: {total_distance:.3f}m")
            
            # Show pick/place info if available
            has_pick = any(h.get('action') == 'pick' for h in position_history)
            has_place = any(h.get('action') == 'place' for h in position_history)
            
            if has_pick or has_place:
                action_str = []
                if has_pick:
                    action_str.append("Picked")
                if has_place:
                    action_str.append("Placed")
                print(f"   Actions: {', '.join(action_str)}")
        
        # Print spatial relations if any
        if 'attributes' in book and 'spatial_relations' in book['attributes']:
            relations = book['attributes']['spatial_relations']
            if relations:
                print(f"   Spatial Relations:")
                for relation in relations:
                    related_id = relation.get('related_to', 'Unknown')
                    related_desc = "Unknown"
                    for n in nodes.values():
                        if n.get('node_id') == related_id:
                            related_desc = n.get('description', 'Unknown')[:30]
                            break
                    
                    rel_type = relation.get('relation_type', 'Unknown')
                    conf = relation.get('confidence', 0)
                    print(f"     - {rel_type} '{related_desc}' (conf: {conf:.2f})")
        print("-" * 40)
    
    print("\n" + "=" * 60)

def main():
    # Default file path
    default_path = "./semantic_forest_save.json"
    
    # Get file path from command line argument or use default
    if len(sys.argv) > 1:
        forest_path = sys.argv[1]
    else:
        forest_path = default_path
    
    # Check if file exists
    if not os.path.exists(forest_path):
        print(f"Error: File {forest_path} does not exist.")
        return
    
    print(f"Opening semantic forest from: {forest_path}")
    
    # Load the semantic forest data
    try:
        with open(forest_path, 'r') as f:
            forest_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse {forest_path}. Invalid JSON format.")
        return
    except Exception as e:
        print(f"Error: Failed to open {forest_path}: {e}")
        return
    
    # Display text summary
    display_forest_details(forest_data)
    
    # Convert to NetworkX graph
    G = convert_forest_to_networkx(forest_data)
    
    # Create visualizations
    try:
        # Create interactive HTML visualization
        plots_dir = Path("./plots")
        plots_dir.mkdir(exist_ok=True)
        
        html_path = plots_dir / "semantic_forest_visualization.html"
        if create_interactive_html(G, str(html_path)):
            print(f"\nInteractive visualization saved to: {html_path}")
            print("Opening in browser...")
            webbrowser.open(f"file://{html_path.absolute()}")
        
        # Create 3D matplotlib visualization with validation
        try:
            fig = create_3d_visualization(G)
            
            # Check if figure has actual data points to display
            has_data = False
            for ax in fig.axes:
                if hasattr(ax, 'collections') and len(ax.collections) > 0:
                    has_data = True
                    break
            
            if has_data:
                plt_path = plots_dir / "semantic_forest_3d.png"
                fig.savefig(plt_path)
                print(f"3D visualization saved to: {plt_path}")
                
                # Show the matplotlib visualization
                plt.show()
            else:
                print("WARNING: No data points to visualize in 3D plot. Skipping figure save.")
                # Create a simple placeholder figure
                plt.figure()
                plt.title("No book data to visualize")
                plt.text(0.5, 0.5, "No book position data available", 
                         ha='center', va='center')
                plt.savefig(plots_dir / "semantic_forest_3d_placeholder.png")
                print("Created placeholder visualization instead.")
        except ValueError as ve:
            print(f"Data validation error in visualization: {ve}")
            print("This error typically occurs when trying to visualize empty datasets.")
            print("Creating a placeholder image instead.")
            
            # Create a simple placeholder figure
            plt.figure()
            plt.title("Visualization Error")
            plt.text(0.5, 0.5, "Error creating 3D visualization", 
                     ha='center', va='center')
            plt.savefig(plots_dir / "semantic_forest_3d_error.png")
        except Exception as e:
            print(f"Unexpected error in 3D visualization: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()