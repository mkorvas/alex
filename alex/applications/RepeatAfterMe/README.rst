RepeatAfterMe (RAM) for Czech - speech data collection
======================================================

This application is useful for bootstraping of speech data. It asks the caller to repeat sentences which are
randomly sampled from a set of preselected sentences.

- The Czech sentences (``sentences_es.txt``) are from Karel Capek novels Matka and RUR, and the Prague's Dependency Treebank.
- The Spanish sentences (``sentences_es.txt``) are taken from the Internet

If you want to run ``ram_hub.py`` on some specific phone number than specify the appropriate extension config:

::

  $ ./ram_hub.py -c ram_hub_LANG.cfg  ../../resources/private/ext-PHONENUMBER.cfg


After collection desired number of calls, use ``copy_wavs_for_transcription.py`` to extract the wave files from
the ``call_logs`` subdirectory for transcription. The files will be copied into into ``RAM-WAVs`` directory.

These calls must be transcribed by the Transcriber or some similar software.