# SNN Digit Lab

Run the app on your Windows computer. It uses your existing Nano 9K model coefficients and the Python simulation of the all-LIF network.

1. Extract the ZIP and open a terminal in the extracted folder.
2. Run `python -m pip install -r requirements.txt` if needed. On Taito's PC, Anaconda already has these packages.
3. Start the server with `python server.py --host 0.0.0.0` or double-click `Start_Digit_Lab.cmd`.
4. Connect the computer and phone to the same trusted Wi-Fi. On the phone, open the printed URL with the computer's Wi-Fi address, including the part after #. The 127.0.0.1 URL is for the computer itself. If Windows asks, allow Python on private networks only. If the detected address belongs to a VPN or virtual adapter, use the Wi-Fi IPv4 address shown by `ipconfig`.
5. Optionally choose an expected digit. Tap **Take a photo**, then accept the photo in the camera. The browser uploads it automatically. **Choose a photo** also works for existing JPEG, PNG, or WebP files.

Each capture is saved in its own folder under `captures` with the original photo, a centered 28×28 image, the actual 8×8 grayscale input, the results JSON, and a single RTL verification vector. Inference runs on the computer. You can use the vector with the existing FPGA testbench to check RTL separately. Spike counts are raw counts over 32 ticks.

Use a single handwritten digit on plain paper and crop close enough to exclude other objects. The model always chooses between 0 and 1. It defaults to 0 when the counts tie. Keep the terminal open while using the app. The phone URL changes when the server restarts. The server uses HTTP on your local network, so use trusted Wi-Fi and do not share the pairing URL or forward the port.
