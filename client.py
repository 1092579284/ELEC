import socket

def main():
    host = 'localhost'
    port = 5001
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    print("Hello, I'm the Oracle. How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        # Send request to server
        client_socket.send(user_input.encode('utf-8'))
        
        # Receive response from server
        response = client_socket.recv(1024).decode('utf-8')
        print(f"Oracle: {response}")
    
    client_socket.close()

if __name__ == "__main__":
    main()
