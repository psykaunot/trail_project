#!/usr/bin/env python3
import time
import threading

def start_link_monitor(locobot):
    """Start a dedicated thread to monitor robot links"""
    def monitor_thread():
        while True:
            print("\n===== ROBOT LINK STATUS =====")
            link_ids = locobot.get_link_ids()
            print(f"Total links: {len(link_ids)}")
            
            for lid in link_ids:
                try:
                    link_name = locobot.get_link_name(lid)
                    node = locobot.get_link_scene_node(lid)
                    position = node.translation
                    rotation = node.rotation
                    print(f"{link_name[:20]:<20} | Pos: {position} | Rot: {rotation.vector}")
                except Exception as e:
                    print(f"Error with link {lid}: {e}")
            
            print("=" * 60)
            time.sleep(5.0)  # Wait 5 seconds between prints
    
    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    print("Link monitor started!")
    return t