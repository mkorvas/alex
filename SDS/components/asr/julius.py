#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import struct
import time
import xml.dom.minidom
import subprocess
import os.path

from os import remove
from tempfile import mkstemp

import __init__

from SDS.components.asr.utterance import *
from SDS.utils.exception import JuliusASRException, JuliusASRTimeoutException
from SDS.utils.various import get_text_from_xml_node

class JuliusASR():
  """ Uses Julius ASR service to recognize recorded audio.

  The main function recognize returns a list of recognised hypotheses.
  One can also obtain a confusion network for the hypotheses.

  """

  def __init__(self, cfg):
    self.recognition_on = False

    self.cfg = cfg
    self.hostname = self.cfg['ASR']['Julius']['hostname']
    self.serverport = self.cfg['ASR']['Julius']['serverport']
    self.adinnetport = self.cfg['ASR']['Julius']['adinnetport']

    try:
      self.cfg['Logging']['system_logger'].debug("Starting the Julius ASR server")
      self.start_server()
      time.sleep(3)
      self.cfg['Logging']['system_logger'].debug("Connecting to the Julius ASR server")
      self.connect_to_server()
      time.sleep(3)
      self.cfg['Logging']['system_logger'].debug("Opening the adinnet connection with the Julius ASR ")
      self.open_adinnet()
    except:
      # always kill the Julius ASR server when there is a problem
      self.julius_server.kill()

  def __del__(self):
    self.julius_server.terminate()
    time.sleep(1)
    self.julius_server.kill()

  def start_server(self):
    jconf = os.path.join(self.cfg['Logging']['system_logger'].output_dir, 'julius.jconf')
    log = os.path.join(self.cfg['Logging']['system_logger'].output_dir, 'julius.log')

    config = open(jconf, "w")
    for k in sorted(self.cfg['ASR']['Julius']['jconf']):
      config.write('%s %s\n' % (k, self.cfg['ASR']['Julius']['jconf'][k]))
    config.close()

    # start the server with the -debug options
    # with this option it does not generates seg faults
    self.julius_server = subprocess.Popen('julius -debug -C %s > %s' % (jconf, log), bufsize = 1, shell=True)

  def connect_to_server(self):
    """Connects to the Julius ASR server to start recognition and receive the recognition oputput."""

    self.s_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s_socket.connect((self.hostname, self.serverport))
    self.s_socket.setblocking(0)

  def open_adinnet(self):
    """Open the audio connection for sending the incoming frames."""

    self.a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.a_socket.connect((self.hostname, self.adinnetport))

  def send_frame(self, frame):
    """Sends one frame of audio data to the Julisus ASR"""

    self.a_socket.sendall(struct.pack("i", len(frame)))
    self.a_socket.sendall(frame)

  def audio_finished(self):
    """"Informs the Julius ASR about the end of segment and that the hypothesis should be finalised."""

    self.a_socket.sendall(struct.pack("i", 0))
    self.recognition_on = False

  def read_audio_command(self):
    """Reads audio command from the Julius adinnet interface.

    Command:
      '0' - pause
      '1' - resume
      '2' - terminate
    """
    self.a_socket.setblocking(0)
    cmd = self.a_socket.recv(1)
    self.socket.a_setblocking(1)
    return cmd

  def read_server_message(self,  timeout = 0.1):
    """Reads a complete message from the Julius ASR server.

    A complete message is denoted by a period on a new line at the end of the string.

    Timeout specifies how long it will wait for the end of message.
    """
    results = ""

    to = 0.0
    while True:
      if to >= timeout:
        raise ASRException("Timeout when waiting for the Julius server message.")

      try:
        results += self.s_socket.recv(1)
      except socket.error:
        # FIX: you should check the type of error. If the server dies then there will be a deadlock

        if not results:
          # there are no data waiting for us
          return None
        else:
          # we already read some data but we did not received the final period, so continue reading
          time.sleep(self.cfg['Hub']['main_loop_sleep_time'])
          to += self.cfg['Hub']['main_loop_sleep_time']
          continue

      if results.endswith("\n.\n"):
        results = results[:-3].strip()
        break

    return results

  def get_results(self, timeout = 0.6):
    """"Waits for the recognition results from the Julius ASR server.

    Timeout specifies how long it will wait for the end of message.
    """
    msg = ""

    # get results from the server
    to = 0.0
    while True:
      if to >= timeout:
        print msg
        raise JuliusASRTimeoutException("Timeout when waiting for the Julius server results.")

      m = self.read_server_message()
      if not m:
        # wait and check whether there is a message
        time.sleep(self.cfg['Hub']['main_loop_sleep_time'])
        to += self.cfg['Hub']['main_loop_sleep_time']
        continue

      msg += m+'\n'

      if '<CONFNET>' in msg:
        break

    if self.cfg['ASR']['Julius']['debug']:
      print msg

    #process the results
    """ Typical result returned by the Julius ASR.

      <STARTPROC/>
      <INPUT STATUS="LISTEN" TIME="1343896296"/>
      <INPUT STATUS="STARTREC" TIME="1343896311"/>
      <STARTRECOG/>
      <INPUT STATUS="ENDREC" TIME="1343896312"/>
      <ENDRECOG/>
      <INPUTPARAM FRAMES="164" MSEC="1640"/>
      <RECOGOUT>
        <SHYPO RANK="1" SCORE="-7250.111328">
          <WHYPO WORD="" CLASSID="<s>" PHONE="sil" CM="0.887"/>
          <WHYPO WORD="I'M" CLASSID="I'M" PHONE="ah m" CM="0.705"/>
          <WHYPO WORD="LOOKING" CLASSID="LOOKING" PHONE="l uh k ih ng" CM="0.992"/>
          <WHYPO WORD="FOR" CLASSID="FOR" PHONE="f er" CM="0.757"/>
          <WHYPO WORD="A" CLASSID="A" PHONE="ah" CM="0.672"/>
          <WHYPO WORD="PUB" CLASSID="PUB" PHONE="p ah b" CM="0.409"/>
          <WHYPO WORD="" CLASSID="</s>" PHONE="sil" CM="1.000"/>
        </SHYPO>
      </RECOGOUT>
      <GRAPHOUT NODENUM="43" ARCNUM="70">
          <NODE GID="0" WORD="" CLASSID="<s>" PHONE="sil" BEGIN="0" END="2"/>
          <NODE GID="1" WORD="" CLASSID="<s>" PHONE="sil" BEGIN="0" END="3"/>
          <NODE GID="2" WORD="" CLASSID="<s>" PHONE="sil" BEGIN="0" END="4"/>
          <NODE GID="3" WORD="I" CLASSID="I" PHONE="ay" BEGIN="3" END="5"/>
          <NODE GID="4" WORD="NO" CLASSID="NO" PHONE="n ow" BEGIN="3" END="7"/>
          <NODE GID="5" WORD="I" CLASSID="I" PHONE="ay" BEGIN="4" END="6"/>
          <NODE GID="6" WORD="UH" CLASSID="UH" PHONE="ah" BEGIN="4" END="6"/>
          <NODE GID="7" WORD="I'M" CLASSID="I'M" PHONE="ay m" BEGIN="4" END="27"/>

          ...

          <NODE GID="38" WORD="PUB" CLASSID="PUB" PHONE="p ah b" BEGIN="79" END="104"/>
          <NODE GID="39" WORD="AH" CLASSID="AH" PHONE="aa" BEGIN="81" END="110"/>
          <NODE GID="40" WORD="LOT" CLASSID="LOT" PHONE="l aa t" BEGIN="81" END="110"/>
          <NODE GID="41" WORD="" CLASSID="</s>" PHONE="sil" BEGIN="105" END="163"/>
          <NODE GID="42" WORD="" CLASSID="</s>" PHONE="sil" BEGIN="111" END="163"/>
          <ARC FROM="0" TO="4"/>
          <ARC FROM="0" TO="3"/>
          <ARC FROM="1" TO="7"/>
          <ARC FROM="1" TO="5"/>
          <ARC FROM="1" TO="6"/>

          ...

          <ARC FROM="38" TO="41"/>
          <ARC FROM="39" TO="42"/>
          <ARC FROM="40" TO="42"/>
      </GRAPHOUT>
      <CONFNET>
        <WORD>
          <ALTERNATIVE PROB="1.000"></ALTERNATIVE>
        </WORD>
        <WORD>
          <ALTERNATIVE PROB="0.950">I</ALTERNATIVE>
          <ALTERNATIVE PROB="0.020">HI</ALTERNATIVE>
          <ALTERNATIVE PROB="0.013">NO</ALTERNATIVE>
          <ALTERNATIVE PROB="0.010"></ALTERNATIVE>
          <ALTERNATIVE PROB="0.006">UH</ALTERNATIVE>
        </WORD>
        <WORD>
          <ALTERNATIVE PROB="0.945">AM</ALTERNATIVE>
          <ALTERNATIVE PROB="0.055">I'M</ALTERNATIVE>
        </WORD>
        <WORD>
          <ALTERNATIVE PROB="1.000">LOOKING</ALTERNATIVE>
        </WORD>
        <WORD>
          <ALTERNATIVE PROB="1.000">FOR</ALTERNATIVE>
        </WORD>
        <WORD>
          <ALTERNATIVE PROB="1.000">A</ALTERNATIVE>
        </WORD>
        <WORD>
          <ALTERNATIVE PROB="0.963">PUB</ALTERNATIVE>
          <ALTERNATIVE PROB="0.016">AH</ALTERNATIVE>
          <ALTERNATIVE PROB="0.012">BAR</ALTERNATIVE>
          <ALTERNATIVE PROB="0.008">LOT</ALTERNATIVE>
        </WORD>
        <WORD>
          <ALTERNATIVE PROB="1.000"></ALTERNATIVE>
        </WORD>
      </CONFNET>
      <INPUT STATUS="LISTEN" TIME="1343896312"/>

    """
    msg = "<RESULTS>"+msg+"</RESULTS>"
    msg = msg.replace("<s>", "&lt;s&gt;").replace("</s>", "&lt;/s&gt;")

    nblist = UtteranceNBList()

    doc = xml.dom.minidom.parseString(msg)
    recogout = doc.getElementsByTagName("RECOGOUT")
    for el in recogout:
      shypo = el.getElementsByTagName("SHYPO")
      for el in shypo:
        whypo = el.getElementsByTagName("WHYPO")
        utterance = ""
        cm = 1.0
        for el in whypo:
          word = el.getAttribute("WORD")
          utterance += " "+word
          if word:
            cm *= float(el.getAttribute("CM"))
        nblist.add(cm, Utterance(utterance))

    nblist.merge()
    nblist.normalise()
    nblist.sort()

    cn = UtteranceConfusionNetwork()

    confnet = doc.getElementsByTagName("CONFNET")
    for el in confnet:
      word = el.getElementsByTagName("WORD")
      for el in word:
        alternative = el.getElementsByTagName("ALTERNATIVE")
        word_list = []
        for el in alternative:
          prob = float(el.getAttribute("PROB"))
          text = get_text_from_xml_node(el)
          word_list.append([prob, text])

        # filter out empty hypotheses
        if len(word_list) == 0:
          continue
        if len(word_list) == 1 and len(word_list[0][1]) == 0:
          continue

        # add the word into the confusion network
        cn.add(word_list)

    cn.merge()
    cn.normalise()
    cn.prune()
    cn.normalise()
    cn.sort()

    return nblist, cn

  def flush(self):
    """Sends command to the Julius ASR to terminate the recognition and get ready for new recognition.
    """

    if self.recognition_on:
      self.audio_finished()

      nblist, cn = self.get_results()
      # read any leftovers
      while True:
        if self.read_server_messages() == None:
          break

    return

  def rec_in(self, frame):
    """ This defines asynchronous interface for speech recognition.

    Call this input function with audio data belonging into one speech segment that should be
    recognized.

    Output hypotheses is obtained by calling hyp_out().
    """

    self.recognition_on = True
    self.send_frame(frame.payload)

    return

  def hyp_out(self):
    """ This defines asynchronous interface for speech recognition.

    Returns recognizers hypotheses about the input speech audio and a confusion network for the input.
    """

    # read all messages accidentally left in the socket from the Julius ASR server before
    # a new ASR hypothesis is decoded
    while True:
      m = self.read_server_message()
      if m == None:
        break

    if self.recognition_on:
      self.audio_finished()

      nblist, cn = self.get_results()

      return cn

    raise JuliusASRException("No ASR hypothesis is available since the recognition has not started.")


