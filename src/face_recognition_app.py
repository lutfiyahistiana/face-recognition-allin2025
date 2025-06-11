import sys # Library untuk sistem operasi dan keluar dari program
from tkinter import * # Library untuk membuat GUI
from tkinter import messagebox # Library untuk menampilkan dialog pesan
from PIL import ImageTk, Image # Library untuk memproses gambar
from tkinter import filedialog # Library untuk dialog pemillihan file
import cv2 # Library OpenCV untuk computer vision dan pemrosesan gambar
import numpy as np # Library NumPy untuk operasi array dan matriks numerik
import os # Library untuk operasi file
import time # Library untuk operasi waktu dan delay
import threading # Library untuk multi-threading
import base64 # Library untuk encoding/decoding base64
import math # Library untuk operasi matematika

def resize_256(image):
    height, width = image.shape[:2] # Mendapatkan tinggi dan lebar gambar
    
    # Jika lebar lebih besar dari tinggi (landscape)
    if width > height:
        start_row = 0 # Mulai dari baris pertama
        end_row = height # Sampai baris terakhir
        # Crop dari tengah secara horizontal
        start_col = (width - height) // 2
        end_col = width - start_col
    else: # Jika tinggi lebih besar dari lebar (portrait)
        start_row = (height - width) // 2
        end_row = height - start_row
        start_col = 0 # Mulai dari kolom pertama
        end_col = width # Sampai kolom terakhir
    
    # Memotong gambar sesuai koordinat yang dihitung
    crop_img = image[start_row:end_row, start_col:end_col]
    dimensi = (256, 256)
    
    # Memilih metode interpolasi berdasarkan ukuran gambar
    if crop_img.shape[0] < 256:
        resized_image = cv2.resize(crop_img, dimensi, interpolation=cv2.INTER_CUBIC)
    else:
        resized_image = cv2.resize(crop_img, dimensi, interpolation=cv2.INTER_AREA)
    
    return resized_image

def get_magnitude(array):
    """
    Fungsi untuk menghitung jarak Euclidean (magnitude/norma) dari array
    Rumus: sqrt(sum(x^2)) untuk semua elemen x dalam array
    """
    return np.sqrt(np.sum(array**2))

def dekomposisi_qr(M):
    """
    Implementasi dekomposisi qr menggunakan householder reflection,
    memecah matriks M menjadi Q (orthogonal) dan R (upper triangular)
    M = Q * R
    """
    rows, cols = np.shape(M) # Mendapatkan dimensi matriks
    Q = np.identity(rows) # Inisialisasi matriks Q sebagai matriks identitas
    R = np.copy(M) # Menyalin matriks M ke R
    
    # Iterasi untuk setiap kolom (kecuali yang terakhir)
    for j in range(rows - 1):
        x = np.copy(R[j:, j]) # Mengambil bagian kolom j dari baris j ke bawah
        x[0] += np.copysign(np.linalg.norm(x), x[0]) # Menambahkan tanda yang sesuai untuk stabilitas numerik
        v = x / np.linalg.norm(x) # Normalisasi vektor untuk mendapatkan vektor Householder
        
        H = np.identity(rows) # Membuat matriks Householder H
        H[j:, j:] -= 2.0 * np.outer(v, v) # Menghitung H = I - 2*v*v^T (Householder reflection)
        
        # Update Q dan R
        Q = Q @ H
        R = H @ R
    
    return Q, np.triu(R) # Mengembalikan Q dan bagian upper triangular dari R

def eigen(M):
    """
    Menghitung eigenvalues dan eigenvectors
    """
    rows, cols = np.shape(M) # Mendapatkan dimensi matriks
    eigVecs = np.identity(rows) # Inisialisasi matriks eigenvector sebagai matriks identitas
    
    # Iterasi QR algorithm (100 iterasi untuk konvergensi)
    for k in range(100):
        s = M.item(rows-1, cols-1) * np.identity(rows) # Shift dengan elemen diagonal kanan bawah untuk mempercepat konvergensi
        Q, R = dekomposisi_qr(np.subtract(M, s)) # QR decomposition dari (M - sI)
        M = np.add(R @ Q, s) # Update M = RQ + sI
        eigVecs = eigVecs @ Q # Akumulasi eigenvectors
    
    return np.diag(M), eigVecs # Eigenvalues ada di diagonal utama, eigenvectors di kolom eigVecs

def get_k_eigen(k, eigenvalues, eigenvectors, filecount):
    """
    Mengambil k eigenvalues dan eigenvectors terbesar
    """
    eigvals = np.array([])
    eigvecs = np.empty((0, filecount), dtype=type(eigenvectors))
    eigvals_copy = eigenvalues.copy() # Untuk manipulasi
    vt = eigenvectors.transpose() # Untuk kemudahan akses
    
    # Mengambil k eigenvalues terbesar
    for i in range(k):
        max_val = eigvals_copy.max()
        eigvals = np.append(eigvals, [max_val])
        
        index = np.where(eigvals_copy == max_val)[0][0]
        eigvals_copy[index] = 0 # Set nilai tersebut ke 0 agar tidak dipilih lagi
        
        vektor = vt[index:index+1] # Mengambil eigenvector yang sesuai
        eigvecs = np.concatenate((eigvecs, vektor), axis=0) # Menambahkan ke array hasil
    
    return eigvals, eigvecs

class VideoCapture:
    def __init__(self, source=0):
        self.source = source # Sumber video
        self.video = None # Objek VideoCapture OpenCV
        self.width = 0 # Lebar frame video
        self.height = 0 # Tinggi frame video
    
    def start(self):
        """
        Memulai capture video
        """
        self.video = cv2.VideoCapture(self.source) # Membuat objek VideoCapture OpenCV
        # Mengecek apakah berhasil membuka sumber video
        if not self.video.isOpened():
            raise RuntimeError("Unable to open video source", self.source)
        # Mendapatkan dimensi frame video
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def stop(self):
        """
        Menghentikan capture video
        """
        # Mengecek apakah video masih aktif
        if self.video is not None and self.video.isOpened():
            self.video.release() # Melepaskan resource kamera
        self.video = None # Reset objek video
    
    def get_image(self):
        """
        Mengambil satu frame dari video
        """
        # Mengecek apakah video masih aktif
        if self.video is not None and self.video.isOpened():
            success, img = self.video.read() # Membaca frame dari video
            if success:
                # Mengkonversi dari BGR (OpenCV) ke RGB (PIL/Tkinter)
                return (success, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return (success, None)
        return (False, None)

class FaceRecognitionModel:
    def __init__(self):
        """
        Inisialisasi model face recognition
        """
        self.path = None # Path dataset untuk training
        self.k = 5  # Jumlah eigenfaces yang digunakan
        self.count = 0 # Jumlah gambar training
        self.data = [] # Data gambar training (flattened)
        self.rawimg = [] # Gambar training asli (untuk display)
        self.filesname = [] # Nama file gambar training
        self.u = None  # Eigenfaces
        self.wdata = None  # Weights dari data training
    
    def train(self, path):
        self.path = path # Set path dataset
        # Reset semua data
        self.count = 0
        self.data = []
        self.rawimg = []
        self.filesname = []
        
        # Inisialisasi vektor untuk menghitung rata-rata
        sum_vector = np.zeros(256 * 256)
        
        # Load dan proses setiap gambar dalam dataset
        for img_name in os.listdir(path):
            # Load gambar dalam grayscale dan color
            img_gray = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_COLOR)
            
            # Resize gambar
            img_gray = resize_256(img_gray)
            img_color = resize_256(img_color)
            
            # Flatten gambar grayscale menjadi vektor 1D
            # Transpose untuk konsistensi dengan konvensi yang digunakan
            img_vector = np.array(img_gray.T).flatten()
            
            # Simpan data
            self.data.append(img_vector) # Vektor gambar untuk training
            self.rawimg.append(img_color) # Gambar asli untuk display
            self.filesname.append(img_name) # Nama file
            
            # Akumulasi untuk menghitung rata-rata
            sum_vector += img_vector
            self.count += 1
        
        mean_vector = sum_vector / self.count # Hitung rata-rata semua gambar training
        
        normalized_data = []
        for img in self.data:
            normalized_data.append(img - mean_vector) # Normalisasi data dengan mengurangi rata-rata
        
        data_matrix = np.array(normalized_data).transpose() # Buat matriks data (N^2 x M) dimana N^2=65536, M=jumlah gambar
        
        # Hitung covariance matrix (M x M) - trick untuk efisiensi komputasi
        # Alih-alih menghitung (N^2 x N^2), kita hitung (M x M) yang lebih kecil
        cov_matrix = np.matmul(data_matrix.transpose(), data_matrix)
        
        # Hitung eigenvalues dan eigenvectors dari covariance matrix
        eigenvalues, eigenvectors = eigen(cov_matrix)
        
        eigvals, eigvecs = get_k_eigen(self.k, eigenvalues, eigenvectors, self.count) # Ambil k eigenvalues dan eigenvectors terbesar
        
        # Hitung eigenfaces (u_i) dari eigenvectors
        # u_i = A * v_i dimana A adalah data matrix, v_i adalah eigenvector
        self.u = np.empty((0, 256*256), dtype=type(eigenvectors))
        for i in range(self.k):
            vi = eigvecs[i].transpose() # Ambil eigenvector ke-i
            ui = np.matmul(data_matrix, vi) # Hitung eigenface
            ui = ui.transpose()
            self.u = np.vstack((self.u, ui)) # Tambahkan ke kumpulan eigenfaces
        
        # Hitung weights untuk setiap gambar training
        # w_i = (gambar - rata-rata) · eigenface_i
        self.wdata = []
        for i in range(self.count):
            wi = [] # Weight untuk gambar ke-i
            for j in range(self.k):
                # Dot product antara gambar normalized dengan eigenface
                weight = np.dot(normalized_data[i], self.u[j])
                wi.append(weight)
            self.wdata.append(wi)
        
        self.wdata = np.array(self.wdata) # Konversi ke numpy array untuk efisiensi
    
    def recognize(self, test_image):
        # Cek apakah model sudah dilatih
        if self.u is None:
            raise ValueError("Model not trained yet")
        
        # Hitung rata-rata dari data training
        mean_vector = np.mean(np.array(self.data), axis=0)
        normalized_test = test_image - mean_vector
        
        # Hitung weights untuk gambar test
        wtest = []
        for j in range(self.k):
            # Dot product dengan setiap eigenface
            weight = np.dot(normalized_test, self.u[j])
            wtest.append(weight)
        wtest = np.array(wtest)
        
        # Hitung jarak Euclidean ke semua gambar training
        distances = []
        for i in range(self.count):
            # Jarak antara weight test dengan weight training ke-i
            distance = get_magnitude(wtest - self.wdata[i])
            distances.append((distance, i))
        
        # Sort berdasarkan jarak (terkecil = paling mirip)
        distances.sort(key=lambda x: x[0])
        
        # Hitung persentase kemiripan
        min_distance = distances[0][0] # Jarak terkecil
        max_distance = distances[-1][0] # Jarak terbesar
        
        # Konversi jarak ke persentase (semakin kecil jarak = semakin tinggi %)
        if max_distance > 0:
            percentage = ((max_distance - min_distance) / max_distance) * 100
        else: # Jika semua jarak sama
            percentage = 100
        
        # Ambil indeks gambar dengan jarak terkecil
        best_match_index = distances[0][1]
        
        # Return hasil recognition
        return {
            'index': best_match_index,
            'filename': self.filesname[best_match_index],
            'image': self.rawimg[best_match_index],
            'percentage': percentage,
            'distance': min_distance,
            'top_3': distances[:3]
        }
    
    def save_cache(self, path):
        """
        Menyimpan model yang sudah dilatih ke file cache agar tidak perlu training ulang
        """
        # Buat direktori cache jika belum ada
        cache_dir = os.path.dirname(path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Simpan semua data model ke file .npz
        np.savez(path, u=self.u, wdata=self.wdata, rawimg=self.rawimg, filesname=self.filesname, count=self.count, data=self.data, k=self.k)
    
    def load_cache(self, dataset_path, cache_path):
        """
        Memuat model dari file cache
        """
        self.path = dataset_path
        # Load data dari file .npz
        data = np.load(cache_path, allow_pickle=True)
        
        # Restore semua variabel model
        self.u = data['u']
        self.wdata = data['wdata']
        self.rawimg = data['rawimg']
        self.filesname = data['filesname']
        self.count = int(data['count'])
        self.data = data['data'].tolist()
        self.k = int(data['k'])

class FaceRecognitionApp:
    """
    Kelas utama untuk aplikasi GUI Face Recognition dengan tkinter
    """
    def __init__(self):
        # Setup window utama
        self.window = Tk()
        self.window.title('Face Recognition') # Judul window
        self.window.geometry("1080x700") # Size
        self.window.configure(bg="#ffffff") # Background color
        self.window.resizable(False, False) # Non-resizable
        
        # Inisialisasi komponen
        self.model = FaceRecognitionModel()
        self.video_capture = VideoCapture()
        
        # Variabel state aplikasi
        self.processing = False # Flag untuk proses recognition
        self.capturing_video = False # Flag untuk video capture
        self.pathdataset = None # Path dataset yang dipilih
        self.imageInput = None # Gambar input untuk recognition
        self.current_result = None # Hasil recognition terkini
        
        # Setup UI dan event handlers
        self.setup_ui()
        self.window.protocol("WM_DELETE_WINDOW", self.on_window_close)
    
    def setup_ui(self):
        # Membuat canvas utama untuk menampung semua elemen UI
        self.canvas = Canvas(
            self.window,
            bg="#ffffff",
            height=700,
            width=1080,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)
        
        # Load dan tampilkan background image (optional, bisa dikomentari jika tidak ada file)
        try:
            self.background_img = PhotoImage(file="assets/appBg.png")
            self.canvas.create_image(540.0, 350.0, image=self.background_img)
        except:
            pass # Jika file background tidak ada, skip
        
        # Buttons
        self.setup_buttons()
        
        # Entries
        self.setup_bg_entries()
        
        # Image display
        self.setup_image_display()
    
    def setup_buttons(self):
        # Tombol choose file dataset
        try:
            self.btnDatasetBg = PhotoImage(file="assets/chooseFile.png")
        except:
            self.btnDatasetBg = None
            
        self.chooseDatasetBtn = Button(
            image=self.btnDatasetBg,
            text="Choose Dataset" if self.btnDatasetBg is None else "",
            borderwidth=0,
            highlightthickness=0,
            command=self.select_dataset,
            relief="flat"
        )
        self.chooseDatasetBtn.place(x=60, y=227, width=150, height=50)
        
        # Tombol choose file image
        try:
            self.btnImgBg = PhotoImage(file="assets/chooseFile.png")
        except:
            self.btnImgBg = None
            
        self.chooseImgBtn = Button(
            image=self.btnImgBg,
            text="Choose Image" if self.btnImgBg is None else "",
            borderwidth=0,
            highlightthickness=0,
            command=self.select_image,
            relief="flat"
        )
        self.chooseImgBtn.place(x=60, y=367, width=150, height=50)
        
        # Tombol start image recognition
        try:
            self.startImgBg = PhotoImage(file="assets/startImgBg.png")
        except:
            self.startImgBg = None
            
        self.startImgBtn = Button(
            image=self.startImgBg,
            text="Start Recognition" if self.startImgBg is None else "",
            borderwidth=0,
            highlightthickness=0,
            command=self.start_recognition_thread,
            relief="flat"
        )
        self.startImgBtn.place(x=50, y=508, width=371, height=70)
        
        # Tombol realtime recognition
        try:
            self.startVideoBg = PhotoImage(file="assets/startVideoBg.png")
            self.stopVideoBg = PhotoImage(file="assets/stopVideoBg.png")
        except:
            self.startVideoBg = None
            self.stopVideoBg = None
            
        self.videoBtn = Button(
            image=self.startVideoBg,
            text="Start Video" if self.startVideoBg is None else "",
            borderwidth=0,
            highlightthickness=0,
            command=self.toggle_video,
            relief="flat"
        )
        self.videoBtn.place(x=50, y=578, width=371, height=70)
    
    def setup_bg_entries(self):
        # Tempat nama dataset yang akan digunakan
        try:
            self.entry1_img = PhotoImage(file="assets/fileChoosedBg.png")
            self.canvas.create_image(324, 252, image=self.entry1_img)
        except:
            pass
            
        self.entry1 = Entry(
            bd=0, bg="#ffffff", highlightthickness=0,
            font=('Poppins', 12, 'bold'), fg='#000000', justify='center'
        )
        self.entry1.place(x=225, y=232, width=198, height=40)
        
        # tempat nama gambar yang akan ditest
        try:
            self.entry2_img = PhotoImage(file="assets/fileChoosedBg.png")
            self.canvas.create_image(324, 392, image=self.entry2_img)
        except:
            pass
            
        self.entry2 = Entry(
            bd=0, bg="#ffffff", highlightthickness=0,
            font=('Poppins', 12, 'bold'), fg='#000000', justify='center'
        )
        self.entry2.place(x=225, y=372, width=198, height=40)

        # Tempat untuk waktu eksekusi
        try:
            self.time_label_img = PhotoImage(file="assets/timeBg.png")
            self.canvas.create_image(671, 538, image=self.time_label_img)
        except:
            pass
            
        self.time_label = Label(
            bd=0, bg="#ffffff", highlightthickness=0,
            font=('Poppins', 12, 'bold'), fg='#000000', justify='center'
        )
        self.time_label.place(x=613, y=518, width=116, height=40)

        # tempat untuk status
        try:
            self.status_label_img = PhotoImage(file="assets/statusBg.png")
            self.canvas.create_image(638, 618, image=self.status_label_img)
        except:
            pass
            
        self.status_label = Label(
            bd=0, bg="#ffffff", highlightthickness=0,
            font=('Poppins', 12, 'bold'), fg='#000000', justify='center'
        )
        self.status_label.place(x=547, y=598, width=182, height=40)

        # Result
        try:
            self.result_label_img = PhotoImage(file="assets/resultBg.png")
            self.canvas.create_image(923, 538, image=self.result_label_img)
        except:
            pass
            
        self.result_label = Label(
            bd=0, bg="#ffffff", highlightthickness=0,
            font=('Poppins', 12, 'bold'), fg='#000000', justify='center'
        )
        self.result_label.place(x=831, y=518, width=184, height=40)

        # Tempat persentase kemiripan
        try:
            self.percentage_label_img = PhotoImage(file="assets/percentBg.png")
            self.canvas.create_image(945, 618, image=self.percentage_label_img)
        except:
            pass
            
        self.percentage_label = Label(
            bd=0, bg="#ffffff", highlightthickness=0,
            font=('Poppins', 12, 'bold'), fg='#000000', justify='center'
        )
        self.percentage_label.place(x=875, y=598, width=140, height=40)
        
        # Inisialisasi entry field dengan placeholder
        self.check_entry()
    
    def setup_image_display(self):
        """
        Setup area untuk menampilkan gambar input
        """
        # Label untuk menampilkan gambar input
        self.left_img_label = Label()
        self.left_img_label.place(x=478, y=227, anchor=NW, width=256, height=256)
        
        # Load gambar default untuk placeholder
        try:
            self.left_img_bg = PhotoImage(file="assets/imgBg.png")
            self.left_img_label.config(image=self.left_img_bg)
            self.left_img_label.image = self.left_img_bg
        except:
            pass
    
    def check_entry(self):
        """
        Mengecek dan mengisi entry field dengan placeholder jika kosong
        """
        if self.entry1.get() == '':
            self.entry1.insert(0, "No File Chosen")
        if self.entry2.get() == '':
            self.entry2.insert(0, "No File Chosen")
    
    def select_dataset(self):
        """
        Fungsi untuk memilih folder dataset training
        """
        # Cegah operasi jika sedang processing atau capturing
        if self.processing or self.capturing_video:
            return
        
        # Buka dialog pemilihan direktori
        path = filedialog.askdirectory()
        if path:
            self.pathdataset = path # Simpan path dataset
            # Update entry field dengan nama folder
            self.entry1.delete(0, END)
            self.entry1.insert(0, os.path.basename(path))
        else: # Jika tidak ada yang dipilih, reset placeholder
            self.check_entry()
    
    def select_image(self):
        """
        Fungsi untuk memilih file gambar untuk recognition
        """
        # Cegah operasi jika sedang processing atau capturing
        if self.processing or self.capturing_video:
            return
        
        # Buka dialog pemilihan file dengan filter gambar
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if path:
            # Update entry field dengan nama file
            self.entry2.delete(0, END)
            self.entry2.insert(0, os.path.basename(path))
            
            # Load dan proses gambar
            image = cv2.imread(path)
            grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # Resize gambar
            resized_image = resize_256(image)
            resized_grayscale = resize_256(grayscale)
            
            # Prepare untuk recognition - flatten ke 1D array
            self.imageInput = np.array(resized_grayscale.T).flatten()
            
            # Display gambar di UI
            display_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            display_image = Image.fromarray(display_image)
            display_image = ImageTk.PhotoImage(display_image)
            
            self.left_img_label.configure(image=display_image)
            self.left_img_label.image = display_image
        else:
            self.check_entry()
    
    def start_recognition_thread(self):
        """Start recognition in separate thread"""
        if self.processing or self.capturing_video:
            return
        
        if self.pathdataset is None or self.imageInput is None:
            messagebox.showerror("Error", "Please select dataset and image")
            return
        
        self.processing = True
        threading.Thread(target=self.start_recognition, daemon=True).start()
        threading.Thread(target=self.counting, daemon=True).start()
    
    def start_recognition(self):
        """Main recognition function"""
        start_time = time.time()
        
        # Cache management
        encoded_path = base64.b64encode(self.pathdataset.encode('ascii')).decode('ascii')
        cache_path = f"_cache_/{encoded_path}.npz"
        
        # Create cache directory if it doesn't exist
        if not os.path.exists("_cache_"):
            os.makedirs("_cache_")
        
        # Load or train model
        if self.model.path != self.pathdataset:
            if os.path.exists(cache_path):
                self.update_status("Loading")
                self.model.load_cache(self.pathdataset, cache_path)
            else:
                self.update_status("Training")
                self.model.train(self.pathdataset)
                self.model.save_cache(cache_path)
        
        self.update_status("Processing")
        
        # Recognize face
        result = self.model.recognize(self.imageInput)
        self.current_result = result
        
        # Display result
        result_image = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_image)
        result_image = ImageTk.PhotoImage(result_image)
        
        # Menampilkan hasil pencocokan
        self.canvas.create_image(764, 227, anchor=NW, image=result_image)
        self.canvas.image = result_image  # Keep reference
        
        # Update labels
        self.result_label.config(text=result['filename'])
        self.percentage_label.config(text=f"{result['percentage']:.1f}%")
        
        # Done
        self.processing = False
        execution_time = time.time() - start_time
        self.time_label.config(text=f"{execution_time:.4f}s")
        self.update_status("Done")
    
    def counting(self):
        start = time.time()
        while self.processing:
            self.time_label.config(text=f"{(time.time() - start):.4f}s")
            time.sleep(0.1)
    
    def toggle_video(self):
        if self.capturing_video:
            self.stop_video()
        else:
            self.start_video()
    
    def start_video(self):
        if self.pathdataset is None:
            messagebox.showerror("Error", "Please choose dataset first")
            return
        
        self.entry2.delete(0, END)
        self.check_entry()
        
        # Change button to stop
        self.videoBtn.config(image=self.stopVideoBg)
        
        self.update_status("Opening")
        self.video_capture.start()
        self.update_status("Capturing")
        self.capturing_video = True
        
        threading.Thread(target=self.video_loop, daemon=True).start()
    
    def video_loop(self):
        start_time = time.time()
        
        while self.capturing_video:
            ret, frame = self.video_capture.get_image()
            if not ret:
                break
            
            resized_image = resize_256(frame)
            countdown = max(0, math.ceil(3 - (time.time() - start_time)))
            
            if countdown > 0:
                cv2.putText(resized_image, str(countdown), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                display_frame = Image.fromarray(resized_image)
                display_frame = ImageTk.PhotoImage(display_frame)
                self.left_img_label.configure(image=display_frame)
                self.left_img_label.image = display_frame
            else:
                # Prepare for recognition
                resized_grayscale = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
                self.imageInput = np.array(resized_grayscale.T).flatten()
                
                # Display frame
                display_frame = Image.fromarray(resized_image)
                display_frame = ImageTk.PhotoImage(display_frame)
                self.left_img_label.configure(image=display_frame)
                self.left_img_label.image = display_frame
                
                # Run recognition
                if not self.processing:
                    self.processing = True
                    threading.Thread(target=self.start_recognition, daemon=True).start()
                    time.sleep(3)  # Wait before next recognition
                    start_time = time.time()
            
            time.sleep(0.1)
    
    def stop_video(self):
        self.capturing_video = False
        self.video_capture.stop()
        
        # Reset display
        self.left_img_label.config(image=self.left_img_bg)
        self.left_img_label.image = self.left_img_bg
        
        # Change button back to start
        self.videoBtn.config(image=self.startVideoBg)
        
        # Clear results
        self.imageInput = None
        self.update_status("")
        self.result_label.config(text="")
        self.percentage_label.config(text="")
    
    def update_status(self, status):
        self.status_label.config(text=status)
    
    def on_window_close(self):
        self.capturing_video = False
        self.video_capture.stop()
        self.window.destroy()
        sys.exit()
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    try:
        app = FaceRecognitionApp()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")