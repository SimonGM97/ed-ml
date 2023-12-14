import psutil

def kill_process_by_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        if proc.info.get('connections', []) is not None:
            for conn in proc.info.get('connections', []):
                if conn.get('status') == 'LISTEN' and conn.get('local_address') and conn['local_address'][1] == port:
                    print(f"Killing process {proc.info['pid']} using port {port}")
                    psutil.Process(proc.info['pid']).terminate()

# Example: Kill process using port 5000
kill_process_by_port(5000)
kill_process_by_port(8501)

# lsof -i :5000
# kill -9 <PID>