import sys
from PyQt5 import Qt
from PyQt5 import QtCore,QtGui
from PyQt5.QtWidgets import QWidget,QMainWindow,QHeaderView, QMessageBox, QFileDialog
from TTS_Layout import Ui_MainWindow
import time
from hyperparams import Hyperparams as hp
import synthesize
from graph import Graph
import requests
import json
import datetime
import textwrap
import numpy as np
import os
import num2words
import re
import pygame

class GUI(QMainWindow, Ui_MainWindow):
    def __init__(self,app):
        super(GUI, self).__init__()
        #self.pushButton.pressed.connect(self.update_weather)
        #self.threadpool = QThreadPool()
              
        self.app = app
        self.setupUi(self)
        self.setWindowTitle("TTS GUI Version %s" %0.1)
        self.ModelCombo.currentIndexChanged.connect(self.selection_change)
        self.TTSDialogButton.clicked.connect(self.start_synthesis)
        self.TTSSkipButton.clicked.connect(self.skip_wav)
        self.TTSSkipButton.setDisabled(True)
                
        self.logs = []
        self.logs2 = []
        self.max_log_lines = 3
        self.model_dir = [] # Stores list of paths
        self.model_name = [] # Stores list of names
        # Load graph
        self.g = Graph(mode="synthesize")
        #print("Graph loaded")
        
        # Because of bug in streamelements timestamp filter, need 2 variables for previous time
        ## TODO Add button for start time of donations
        self.startup_time = datetime.datetime.utcnow().isoformat()
        #self.startup_time = '0' # For debugging
        self.prev_time = datetime.datetime.utcnow().isoformat() 
        #self.prev_time = '0' # for debugging
        self.offset = 0
        
        self.ClientStartBtn.clicked.connect(self.start_client)
        self.ClientSkipBtn.clicked.connect(self.skip_wav)
        self.channel_id = ''
        self.ClientStopBtn.clicked.connect(lambda: self.set_client_flag(False))
        self.client_flag = False
        self.ClientStopBtn.setDisabled(True)
        self.ClientSkipBtn.setDisabled(True)
        self.ModelCombo.setDisabled(True)
        self.TTSDialogButton.setDisabled(True)
        self.LoadTTSButton.clicked.connect(self.add_model_path)
        self.tab_2.setDisabled(True)
        self.log_window2.ensureCursorVisible()
        self.update_log_window("Begin by loading a model")
        pygame.mixer.quit()
        pygame.mixer.init(frequency=22050,size=-16, channels=1)
        self.channel = pygame.mixer.Channel(0)

    def startup_update(self):
        if not self.ModelCombo.isEnabled():
            self.ModelCombo.setEnabled(True)    
        if not self.tab_2.isEnabled():
            self.tab_2.setEnabled(True)
        if not self.TTSDialogButton.isEnabled():
            self.TTSDialogButton.setEnabled(True)

    def add_model_path(self):
        fpath = self.open_folder_dialog()
        fpath = fpath[:-2]# Remove the -1 -2 naming standard
        if fpath not in self.model_dir:
            head,tail = os.path.split(fpath) # Split into parent and child dir
            self.model_dir.append(fpath) # Save full path
            self.model_name.append(tail) # Save name of model
            combo_idx = self.ModelCombo.count()
            self.ModelCombo.addItem(tail)
            self.startup_update()
            self.update_log_window("Added "+tail)


    def open_folder_dialog(self):
        dialog = QFileDialog()
        fpath = dialog.getExistingDirectory(self, 'Select pretrained model directory')
        return fpath

    def get_current_model_dir(self):
        return self.model_dir[self.ModelCombo.currentIndex()]

    def selection_change(self,i):
        self.model_selected = self.model_name[i]
        #print("Current index",i,"selection changed ",self.model_name[i])

    def update_log_bar(self,val,maximum=1):
        self.progressBar.setValue(val * 100)
        self.progressBar.setMaximum(maximum * 100)
        self.progressBar.setTextVisible(val != 0)
        self.app.processEvents()
    def update_log_bar2(self,val,maximum=1):
        self.progressBar2.setValue(val * 100)
        self.progressBar2.setMaximum(maximum * 100)
        self.progressBar2.setTextVisible(val != 0)
        self.app.processEvents()

    def update_log_window(self, line, mode="newline"):
        if mode == "newline" or not self.logs:
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = '\n'.join(self.logs)
        
        self.log_window1.setText(log_text)
        self.app.processEvents()

    def get_TTSwindow_text(self):
        input = self.TTSTextEdit.toPlainText()
        lines = textwrap.wrap(input, hp.max_N-1, break_long_words=False) 
        if any(np.array([len(i) for i in lines])>hp.max_N-1): 
            return (None,False)
        else:
            return (input,True)
    def playback_wav(self,wav):
        if self.tabWidget.currentIndex()==0:
            self.TTSSkipButton.setEnabled(True)
        else:
            self.ClientSkipBtn.setEnabled(True)
        sound = pygame.mixer.Sound(wav)
        self.channel.queue(sound)

    def skip_wav(self):
        if self.channel.get_busy():
            self.channel.stop()
        self.TTSSkipButton.setDisabled(True)
        self.ClientSkipBtn.setDisabled(True)

    def start_synthesis(self):
        self.update_status_bar("Analyzing Text")
        (text,valid) = self.get_TTSwindow_text()
        if not valid:
            self.update_log_window('Word is too long, break it up with spaces')
            return
        text = self.pre_process_str(text)
        self.TTSDialogButton.setDisabled(True)
        self.ModelCombo.setDisabled(True)
        self.TTSTextEdit.setDisabled(True)
        self.LoadTTSButton.setDisabled(True)
        self.update_log_bar(0)
        self.update_log_window('Using TTS Voice: '+self.model_selected)
        self.update_log_window('Initializing')
        self.update_status_bar("Creating voice")
        def synthesis_progress(j,elapsed):
            self.update_log_bar(j/hp.max_T)
            self.update_log_window('Elapsed: '+str(int(elapsed))+'s','overwrite')
        ## TODO fix model path at run
        ## TODO add number to word conversion
        model_fpath = self.get_current_model_dir()
        wav = synthesize.synthesize(text, model_fpath, self.g, synthesis_progress)
        self.playback_wav(wav)
        self.update_log_window('Done')
        self.TTSDialogButton.setEnabled(True)
        self.ModelCombo.setEnabled(True)
        self.TTSTextEdit.setEnabled(True)
        self.LoadTTSButton.setEnabled(True)
        self.update_status_bar("Ready")
        
    def update_log_window_2(self, line, mode="newline"):
        if mode == "newline" or not self.logs2:
            self.logs2.append(line)
        elif mode == "append":
            self.logs2[-1] += line
        elif mode == "overwrite":
            self.logs2[-1] = line
        log_text = '\n'.join(self.logs2)
        
        self.log_window2.setText(log_text)
        self.log_window2.verticalScrollBar().setValue(
            self.log_window2.verticalScrollBar().maximum())
        self.app.processEvents()

    def update_status_bar(self, line):
        self.statusbar.setText(line)
        self.app.processEvents()


    def get_token(self):
        TOKEN = ''.join(self.APIKeyLine.text().split())
        return TOKEN

    def set_client_flag(self,val):
        self.client_flag = val
   
    def validate_se(self):
        # Connect to streamelement and saves channel id
        # return true if chn id and token returns valid
        # Test Channel ID
        self.update_status_bar("Validating StreamElements")
        CHANNEL_NAME = ''.join(self.ChannelName.text().split())
        url = "https://api.streamelements.com/kappa/v2/channels/"+CHANNEL_NAME
        response = requests.request("GET", url, headers={'accept': 'application/json'})
        if response.status_code == 200:
            # Test JWT Token
            self.channel_id = json.loads(response.text)['_id']
            url = "https://api.streamelements.com/kappa/v2/tips/"+self.channel_id
            querystring = {"offset":"0","limit":"10","sort":"createdAt","after":"0","before":"0"}
            TOKEN = self.get_token()
            headers = {'accept': 'application/json',"Authorization": "Bearer "+TOKEN}
            response2 = requests.request("GET", url, headers=headers, params=querystring)
            if response2.status_code == 200:
                self.update_log_window_2("\nConnected to "+CHANNEL_NAME)
                self.set_client_flag(True)
                return True
            else: 
                self.update_log_window_2("\nDouble check your token")
                print(response2.text)
        else: 
            self.update_log_window_2("\nDouble check your channel name")
            print(response.text)
        
        return False

    def start_client(self):
        # TODO rewrite for multithreading

        # TODO rewrite for websockets (need to set up queue and thread first)
        if not self.validate_se():
            return
        self.ClientStartBtn.setDisabled(True)
        self.ClientStopBtn.setEnabled(True)
        self.ClientSkipBtn.setEnabled(True)
        
        self.tab.setDisabled(True)
        self.ClientAmountLine.setDisabled(True)
        self.update_status_bar("Connecting to StreamElements")
        url = "https://api.streamelements.com/kappa/v2/tips/"+self.channel_id
        headers = {'accept': 'application/json',"Authorization": "Bearer "+self.get_token()}
        self.update_log_window_2('Using TTS Voice: '+self.model_selected)
        self.update_log_window_2('Initializing')
        min_donation = self.get_min_donation()
        self.update_log_window_2('Minimum amount for TTS: '+str(min_donation))
        while self.client_flag:
            if not self.channel.get_busy():
                # TODO add skip button (needs threading)
                print('Polling', datetime.datetime.utcnow().isoformat())
                self.update_status_bar("Waiting for incoming donations")
                low_limit = 0 # change to current_day - 1 for midnight spill over
                current_time = datetime.datetime.utcnow().isoformat()
                querystring = {"offset":self.offset,"limit":"10","sort":"createdAt","after":self.startup_time,"before":current_time}
                response = requests.request("GET", url, headers=headers, params=querystring)
                #print(response.text)
                data = json.loads(response.text)
                for count,dono in enumerate(data['docs']):
                    self.update_status_bar("Processing donations")
                    dono_time = dono['createdAt']
                    self.offset += 1
                    if dono_time > self.prev_time:
                        amount = dono['donation']['amount'] # Int
                        if float(amount) >= min_donation:
                            name = dono['donation']['user']['username']
                            msg = dono['donation']['message']
                            ## TODO Allow multiple speaker in msg using [] or ()
                            msg = self.pre_process_str(msg)
                            currency = dono['donation']['currency']
                            dono_id = dono['_id']
                    
                            self.update_log_window_2('\n###########################')
                            self.update_log_window_2(name+' donated '+currency+str(amount))
                            self.update_log_window_2(msg)
                            # send to TTS
                            def synthesis_progress(j,elapsed):
                                self.update_log_bar2(j/hp.max_T)
            
                            model_fpath = self.get_current_model_dir()
                            wav=synthesize.synthesize(msg, model_fpath, self.g, synthesis_progress)
                            
                            self.playback_wav(wav)
                            # Increment time
                            self.prev_time = dono_time
                            
            time.sleep(0.1)
            self.app.processEvents()
        print("end")
        self.ClientStartBtn.setEnabled(True)
        self.ClientStopBtn.setDisabled(True)
        self.ClientSkipBtn.setDisabled(True)
        self.tab.setEnabled(True)
        self.ClientAmountLine.setEnabled(True)
        self.update_log_window_2('\nDisconnected')
        self.update_status_bar("Ready")

    def pre_process_str(self,msg):
        # Replaces digits with words
        msg = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), msg)
        return msg
    
    def get_min_donation(self):
        return float(self.ClientAmountLine.value())


    
#class MyThread(QThread):
#    # Create a counter thread
#    change_value = pyqtSignal(int)
#    def run(self):
#        cnt = 0
#        while cnt &lt; 100:
#            cnt+=1
#            time.sleep(0.3)
#            self.change_value.emit(cnt)    

if __name__ == '__main__':
    #app = QApplication([])
    #window = MainWindow()
    #app.exec_()
    app = Qt.QApplication(sys.argv)
    window = GUI(app)
    window.show()
    sys.exit(app.exec_())