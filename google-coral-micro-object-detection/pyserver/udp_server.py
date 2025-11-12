# udp_server.py
import socket
import json
import time
import struct

# Define the binary packet structure (must match MCU exactly)
BINARY_PACKET_FORMAT = '<IIIII32s' + 'fffffB' * 60  # Little-endian format
BINARY_PACKET_SIZE = struct.calcsize(BINARY_PACKET_FORMAT)

def decode_binary_packet(data):
    """Decode binary packet from MCU"""
    try:
        if len(data) != BINARY_PACKET_SIZE:
            print(f"WARNING: Expected {BINARY_PACKET_SIZE} bytes, got {len(data)}")
            return None
            
        # Unpack the binary data
        unpacked = struct.unpack(BINARY_PACKET_FORMAT, data)
        
        # Extract header fields
        msg_id = unpacked[0]
        total_expected = unpacked[1]
        dtime = unpacked[2]
        num_bboxes = unpacked[3]
        payload_size = unpacked[4]
        image_filename = unpacked[5].decode('utf-8').rstrip('\x00')
        
        # Extract bounding boxes
        bboxes = []
        offset = 6
        for i in range(min(num_bboxes, 100)):  # max 100 bboxes
            bbox_data = unpacked[offset:offset+6]
            bboxes.append({
                'xmin': bbox_data[0],
                'ymin': bbox_data[1], 
                'xmax': bbox_data[2],
                'ymax': bbox_data[3],
                'score': bbox_data[4],
                'id': bbox_data[5]
            })
            offset += 6
            
        return {
            'msg_id': msg_id,
            'total_expected': total_expected,
            'image': image_filename,
            'dtime': dtime,
            'num_bboxes': num_bboxes,
            'payload_size': payload_size,
            'bboxes': bboxes
        }
    
    except Exception as e:
        print(f"Error decoding binary packet: {e}")
        return None

def udp_receiver_binary():
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"Listening for binary UDP packets on {UDP_IP}:{UDP_PORT}")
    print(f"Expected packet size: {BINARY_PACKET_SIZE} bytes")
    
    received_msgs = {}
    total_received = 0
    last_packet_time = None
    
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            total_received += 1
            receive_timestamp = time.time() * 1000  # ms
            
            # Calculate inter-packet interval
            inter_packet_time = None
            if last_packet_time is not None:
                inter_packet_time = receive_timestamp - last_packet_time
            last_packet_time = receive_timestamp

            packet = decode_binary_packet(data)
            if packet is None:
                print(f"Failed to decode packet from {addr}")
                continue
            
            msg_id = packet['msg_id']
            image_name = packet['image']
            dtime = packet['dtime']
            num_bboxes = packet['num_bboxes']
            total_expected = packet['total_expected']
            payload_size = packet['payload_size']
            bboxes = packet['bboxes']

            received_msgs[msg_id] = {
                'image': image_name,
                'receive_timestamp': receive_timestamp,
                'inter_packet_time': inter_packet_time,
                'dtime': dtime,
                'num_bboxes': num_bboxes,
                'payload_size': payload_size,
                'total_expected': total_expected,
                'bboxes': bboxes,
                'received_at': receive_timestamp
            }
            
            if inter_packet_time:
                print("=" * 50)
                print(f" Binary msg_id={msg_id}, image={image_name}, dtime={dtime}ms, bboxes={num_bboxes}, "
                      f"interval={inter_packet_time:.2f}ms, size={len(data)}B, total_expected={total_expected}")
            else:
                print("=" * 50)
                print(f" Binary msg_id={msg_id}, image={image_name}, dtime={dtime}ms, bboxes={num_bboxes}, "
                      f"(first packet), size={len(data)}B, total_expected={total_expected}")
            
            

            # Print individual bboxes
            for i, bbox in enumerate(bboxes):
                print(f"  bbox {i}: id={bbox['id']}, score={bbox['score']:.2f}, "
                      f"xmin={bbox['xmin']:.1f}, ymin={bbox['ymin']:.1f}, "
                      f"xmax={bbox['xmax']:.1f}, ymax={bbox['ymax']:.1f}")


    except KeyboardInterrupt:
        print(f"\nReceived {total_received} total messages")
        print(f"Unique message IDs: {len(received_msgs)}")
        
        if received_msgs:
            # Calculate packet loss
            total_expected = list(received_msgs.values())[0].get('total_expected', 0)
            if total_expected > 0:
                missing_msgs = []
                for i in range(1, total_expected + 1):
                    if i not in received_msgs:
                        missing_msgs.append(i)

                if missing_msgs:
                    print(f"Missing messages: {missing_msgs}")
                    loss_rate = (len(missing_msgs) / total_expected) * 100
                    print(f"Packet loss rate: {loss_rate:.2f}%")
                else:
                    print(" All messages received successfully!")
            
            # Calculate statistics
            inter_packet_times = [msg['inter_packet_time'] for msg in received_msgs.values() 
                                 if msg['inter_packet_time'] is not None]
            dtimes = [msg['dtime'] for msg in received_msgs.values()]
            
            if inter_packet_times:
                avg_interval = sum(inter_packet_times) / len(inter_packet_times)
                min_interval = min(inter_packet_times)
                max_interval = max(inter_packet_times)
                
            avg_dtime = sum(dtimes) / len(dtimes)
            total_bboxes = sum(msg['num_bboxes'] for msg in received_msgs.values())
            
            print(f"\n=== BINARY PROTOCOL STATISTICS ===")
            if inter_packet_times:
                print(f"Avg inter-packet time: {avg_interval:.2f} ms")
                print(f"Min/Max inter-packet: {min_interval:.2f} / {max_interval:.2f} ms")
            print(f"Avg processing time (dtime): {avg_dtime:.2f} ms")
            print(f"Total bboxes detected: {total_bboxes}")
            print(f"Packet size: {BINARY_PACKET_SIZE} bytes (vs ~600-800 JSON)")
            print(f"Bandwidth savings: ~75-80%")
            print(f"===================================")
        
        sock.close()
        print("UDP server closed.")


def udp_receiver():
    UDP_IP = "0.0.0.0"  # Listen on all interfaces
    UDP_PORT = 5005
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")
    
    received_msgs = {}
    total_received = 0
    last_packet_time = None
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            total_received += 1
            receive_timestamp = time.time() * 1000  # ms

            # Calculate inter-packet interval
            inter_packet_time = None
            if last_packet_time is not None:
                inter_packet_time = receive_timestamp - last_packet_time
            last_packet_time = receive_timestamp

            try:
                raw_data = data.decode('utf-8')
                msg = json.loads(raw_data)

                msg_id = msg.get('msg_id', 0)
                image_name = msg.get('image', 'unknown')
                dtime = msg.get('dtime', 0)
                num_bboxes = msg.get('num_bboxes', 0)
                payload_size = msg.get('payload_size', 0)
                total_expected = msg.get('total_expected', 0)  # Add this line!
                bboxes = msg.get('bboxes', [])
                
                received_msgs[msg_id] = {
                    'image': image_name,
                    'receive_timestamp': receive_timestamp,
                    'inter_packet_time': inter_packet_time,
                    'dtime': dtime,
                    'num_bboxes': num_bboxes,
                    'payload_size': payload_size,
                    'total_expected': total_expected,
                    'bboxes': bboxes,  # Store the actual bounding boxes
                    'received_at': receive_timestamp
                }
                
                if inter_packet_time:
                    print(f"Received msg_id={msg_id}, image={image_name}, dtime={dtime}ms, bboxes={num_bboxes}, "
                          f"interval_since_last={inter_packet_time:.2f}ms, total_expected={total_expected}")
                else:
                    print(f"Received msg_id={msg_id}, image={image_name}, dtime={dtime}ms, bboxes={num_bboxes}, "
                          f"(first packet), total_expected={total_expected}")

            except json.JSONDecodeError as e:
                print(f"Invalid JSON from {addr}")
                print(f"Error: {e}")
                print(f"Raw data: {data.decode('utf-8', errors='replace')}")
                
    except KeyboardInterrupt:
        print(f"\nReceived {total_received} total messages")
        print(f"Unique message IDs: {len(received_msgs)}")
        

        if received_msgs:
            total_expected = list(received_msgs.values())[0].get('total_expected', 0)
            if total_expected > 0:
                print(f"Total expected messages: {total_expected}")
            
            missing_msgs = []
            for i in range(1, total_expected + 1):  # Assuming IDs start from 1
                if i not in received_msgs:
                    missing_msgs.append(i)

            if missing_msgs:
                print(f"Missing messages: {missing_msgs}")
                print(f"Total missing: {len(missing_msgs)}")
                loss_rate = (len(missing_msgs) / total_expected) * 100
                print(f"True packet loss rate: {loss_rate:.2f}%")
            else:
                print("✅ All messages received successfully!")
            
            # Calculate inter-packet statistics
            inter_packet_times = [msg['inter_packet_time'] for msg in received_msgs.values() if msg['inter_packet_time'] is not None]
            if inter_packet_times:
                avg_interval = sum(inter_packet_times) / len(inter_packet_times)
                min_interval = min(inter_packet_times)
                max_interval = max(inter_packet_times)
            
            # Processing time stats
            dtimes = [msg['dtime'] for msg in received_msgs.values()]
            avg_dtime = sum(dtimes) / len(dtimes)
            
            # Payload stats
            total_bboxes = sum(msg['num_bboxes'] for msg in received_msgs.values())
            avg_payload = sum(msg['payload_size'] for msg in received_msgs.values()) / len(received_msgs)
            
            print(f"\n=== SUMMARY STATISTICS ===")
            if inter_packet_times:
                print(f"Avg inter-packet time: {avg_interval:.2f} ms")
                print(f"Min/Max inter-packet: {min_interval:.2f} / {max_interval:.2f} ms")
            print(f"Avg processing time (dtime): {avg_dtime:.2f} ms")
            print(f"Total bboxes detected: {total_bboxes}")
            print(f"Avg payload size: {avg_payload:.1f} bytes")
            print(f"==========================")

        sock.close()
        print("UDP server closed.")
    
if __name__ == "__main__":
    # udp_receiver()
    udp_receiver_binary()