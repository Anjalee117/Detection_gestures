"""Desktop regression with a simulated camera; run directly, not via pytest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from utils.preview import VideoPreview
from utils.workers import CameraWorker
from types import SimpleNamespace
from queue import Queue
original=app.tk.Tk
errors=[];checks=[];captures=[]
class Capture:
 def __init__(self):self.released=False;captures.append(self)
 def isOpened(self):return True
 def set(self,*a):pass
 def read(self):return True,app.np.full((480,640,3),(10,30,240),dtype=app.np.uint8)
 def release(self):self.released=True
class Processor:
 def __init__(self,gate):self.gate=gate
 def process(self,image):self.gate.observe('B',.98);return image
 def close(self):pass
class Worker(CameraWorker):
 def __init__(self,factory,gate,**kwargs):super().__init__(lambda:Processor(gate),gate,capture_factory=lambda _:Capture(),**kwargs)
app.CameraWorker=Worker
app.SpeechWorker=lambda:SimpleNamespace(start=lambda:None,stop=lambda:None,join=lambda **kw:None,events=Queue())
def walk(w):
 yield w
 for c in w.winfo_children():yield from walk(c)
def create_root():
 root=original()
 def button(prefix):return next(w for w in walk(root) if isinstance(w,app.tk.Button) and w.cget('text').startswith(prefix))
 def sentence():return next(w for w in walk(root) if isinstance(w,app.tk.Text))
 def check_controls():
  preview=next(w for w in walk(root) if isinstance(w,VideoPreview))
  assert preview.rendered_frames>10
  color=root.tk.call(str(preview._photo),'get',preview._photo.width()//2,preview._photo.height()//2)
  assert tuple(map(int,root.tk.splitlist(color)))==(240,30,10)
  button('CAPTURE LETTER').invoke();assert sentence().get('1.0','end-1c')=='B'
  button('INSERT SPACE').invoke();assert sentence().get('1.0','end-1c')=='B '
  button('BACKSPACE').invoke();assert sentence().get('1.0','end-1c')=='B'
  button('CLEAR SENTENCE').invoke();assert sentence().get('1.0','end-1c')==''
  root.geometry('940x660');root.after(400,check_small)
  checks.append('pixels/capture/space/delete/clear')
 def check_small():
  for prefix in ['CAPTURE LETTER','INSERT SPACE','BACKSPACE','CLEAR SENTENCE','SPEAK SENTENCE','STOP CAMERA']:
   w=button(prefix);assert w.winfo_ismapped(),prefix
   assert w.winfo_rooty()+w.winfo_height()<=root.winfo_rooty()+root.winfo_height(),prefix
  button('STOP CAMERA').invoke();root.after(300,restart)
  checks.append('minimum-size controls')
 def restart():
  assert all(c.released for c in captures)
  button('SELECT / RETRY').invoke();root.after(600,finish)
 def finish():
  assert len(captures)>=2
  preview=next(w for w in walk(root) if isinstance(w,VideoPreview))
  assert preview._photo is not None
  button('STOP CAMERA').invoke()
  checks.append('stop/restart')
  print('GUI REGRESSION PASS',checks,flush=True)
  root.destroy()
 def error(*args):errors.append(str(args[1]));root.destroy()
 root.after(50,lambda:setattr(root,'report_callback_exception',error))
 root.after(500,lambda:button('OPEN DASHBOARD').invoke())
 root.after(3500,check_controls);root.after(7000,root.destroy)
 return root
app.tk.Tk=create_root
app.main()
assert len(checks)==3 and not errors,(checks,errors)
