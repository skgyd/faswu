import sys
from PyQt5.QtWidgets import *

class Connect(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Connect')
        self.setGeometry(100, 200, 200, 200)

        layout = QVBoxLayout()

        ipLabel = QLabel('IP:')
        ip = QLineEdit()
        font = ip.font()
        font.setPointSize(11)
        ip.setFont(font)
        self.ip = ip

        portLabel = QLabel('port:')
        port = QLineEdit()
        font = port.font()
        font.setPointSize(11)
        port.setFont(font)
        self.port = port

        subLayout = QHBoxLayout()

        btnServer = QPushButton("Server")
        btnServer.clicked.connect(self.server)

        btnClient = QPushButton("Client")
        btnClient.clicked.connect(self.client)

        layout.addWidget(ipLabel)
        layout.addWidget(ip)
        layout.addWidget(portLabel)
        layout.addWidget(port)
        layout.addStretch(1)

        subLayout.addWidget(btnServer)
        subLayout.addWidget(btnClient)
        layout.addLayout(subLayout)

        self.setLayout(layout)

    def server(self):
        self.con = 1
        self.accept()

    def client(self):
        self.con = 2
        self.accept()


    def showModal(self):
        return super().exec_()