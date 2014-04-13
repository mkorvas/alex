#!/usr/bin/env python
# vim: set fileencoding=utf-8 fdm=marker :
"""
This module provides tools for **ENGLISH** normalisation of transcriptions, mainly for
those obtained from human transcribers.
"""

from __future__ import unicode_literals

import re

__all__ = ['normalise_text', 'exclude', 'exclude_by_dict']

_nonspeech_events = ['_SIL_', '_INHALE_', '_LAUGH_', '_EHM_HMM_', '_NOISE_', '_EXCLUDE_',]

for idx, ne in enumerate(_nonspeech_events):
    _nonspeech_events[idx] = (re.compile(r'((\b|\s){pat}(\b|\s))+'.format(pat=ne)), ' '+ne+' ')

# nonspeech event transcriptions {{{
_nonspeech_map = {
    '_SIL_': (
        '(SIL)',
        '(QUIET)',
        '(CLEARING)'),
    '_INHALE_': (
        '(INHALE)',
        '(BREATH)',
        '(BREATHING)',
        '(SNIFFING)'),
    '_LAUGH_': (
        '(LAUGH)',
        '(LAUGHING)'),
    '_EHM_HMM_': (
        '(EHM_HMM)',
        '(HESITATION)',
        '(HUM)'),
    '_NOISE_': (
        '(COUCHING)',
        '(COUGH)',
        '(COUGHING)',
        '(LIPSMACK)',
        '(POUNDING)',
        '(RING)',
        '(RINGING)',
        '(INTERFERENCE)',
        '(KNOCKING)',
        '(BANG)',
        '(BANGING)',
        '(BACKGROUNDNOISE)',
        '(BABY)',
        '(BARK)',
        '(BARKING)',
        '(NOISE)',
        '(NOISES)',
        '(SCRAPE)',
        '(STATIC)',
        '(SQUEAK)',
        '(TVNOISE)'
    ),
    '_EXCLUDE_': (
        '(EXCLUDE)',
        '(UNINTELLIGIBLE)',
        '(UNINT)',
        '(PERSONAL)',
        '(VULGARISM)',
    )
}
#}}}
_nonspeech_trl = dict()
for uscored, forms in _nonspeech_map.iteritems():
    for form in forms:
        _nonspeech_trl[form] = uscored

# substitutions {{{
_subst = [
          ('UNINTELLIGIBLE', '_EXCLUDE_'),
          ('UNINT', '_EXCLUDE_'),
          ('11', 'ELEVEN'),
          ('24', 'TWENTY FOUR'),
          ('3', 'THREE'),
          ('4', 'FOUR'),
          ('5', 'FIVE'),
          ('73', 'SEVENTY THREE'),
          ('\'BOUT', 'ABOUT'),
          ('\'M', '_EXCLUDE_'),
          ('ABUT', 'ABOUT'),
          ('ABOTU', 'ABOUT'),
          ('ACCOMIDATION', 'ACCOMMODATION'),
          ('ACCOMODATION', 'ACCOMMODATION'),
          ('ACONTEMPORARY', 'A CONTEMPORARY'),
          ('ADAMBROOKS', 'ADDENBROOKE\'S'),
          ('ADDDRESS', 'ADDRESS'),
          ('ADDEENBROOKE\'S', 'ADDENBROOKE\'S'),
          ('ADENBOOKS', 'ADDENBROOKE\'S'),
          ('ADDELBROOKE\'S', 'ADDENBROOKE\'S'),
          ('ADDENBOOKE', 'ADDENBROOKE'),
          ('EDENBROOKE', 'ADDENBROOKE'),
          ('ADDENBROOKE\'S\'A', 'ADDENBROOKE\'S'),
          ('ADDENBROOKES\'A',  'ADDENBROOKE\'S'),
          ('ADDENBROOKE\'S\'',  'ADDENBROOKE\'S'),
          ('ADDENBROOKES', 'ADDENBROOKE\'S'),
          ('ADDENBROOKES\'',  'ADDENBROOKE\'S'),
          ('ADDENBROOKE\'S\'S', 'ADDENBROOKE\'S'),
          ('ADDENBROOKS', 'ADDENBROOKE\'S'),
          ('ADENBROOKES', 'ADDENBROOKE\'S'),
          ('ADDENSBROOKE', 'ADDENBROOKE\'S'),
          ('ADDENBROOKE`S', 'ADDENBROOKE\'S'),
          ('ADDENBROOK\'S', 'ADDENBROOKE\'S'),
          ('ADDENBROOK', 'ADDENBROOKE'),
          ('ADENBROOKE', 'ADDENBROOKE'),
          ('ADDERSS', 'ADDRESS'),
          ('ADDESS', 'ADDRESS'),
          ('ADDNEBROOKE', 'ADDENBROOKE\'S'),
          ('ADDNEBROOKE\'S',  'ADDENBROOKE\'S'),
          ('ADDRES', 'ADDRESS'),
          ('ADDRFESS', 'ADDRESS'),
          ('ADDRESSOF', 'ADDRESS OF'),
          ('ADDRESSPHONE', 'ADDRESS PHONE'),
          ('ADDRESSS', 'ADDRESS'),
          ('ADENBROOKE\'S', 'ADDENBROOKE\'S'),
          ('ADENBROOK\'S', 'ADDENBROOKE\'S'),
          ('ADENBROOKS', 'ADDENBROOKE\'S'),
          ('ALLRIGHT', 'ALRIGHT'),
          ('AD', '_EXCLUDE_'),
          ('ADRDESS', 'ADDRESS'),
          ('ADRESS', 'ADDRESS'),
          ('ADRESSES', 'ADDRESSES'),
          ('ADRESSS', 'ADDRESS'),
          ('ADSRESSSIR', 'ADDRESS SIR'),
          ('AFORDABLE', 'AFFORDABLE'),
          ('AIDENBROOK', 'ADDENBROOKE\'S'),
          ('AMARICAN', 'AMERICAN'),
          ('ANDAND', 'AND AND'),
          ('ANDAREA', 'AND AREA'),
          ('ANDCHINESE', 'AND CHINESE'),
          ('ANDENGLISH', 'AND ENGLISH'),
          ('ANDTHAT', 'AND THAT'),
          ('ANDWELL', 'AND WELL'),
          ('ANDWHAT', 'AND WHAT'),
          ('ANDWHERE', 'AND WHERE'),
          ('ANEXPENSIVE', 'AN EXPENSIVE'),
          ('ANYKIND', 'ANY KIND'),
          ('ANYTIHNG', 'ANYTHING'),
          ('ANYWHERE\'S FINE', 'ANYWHERE IS FINE'),
          ('APUB', 'A PUB'),
          ('ARBORY', 'ARBURY'),
          ('ARCHECTICTURE', 'ARCHITECTURE'),
          ('AREA`', 'AREA'),
          ('ARESTAURANT', 'A RESTAURANT'),
          ('ATHAI', 'A THAI'),
          ('ATURKISH', 'A TURKISH'),
          ('AVANUE', 'AVENUE'),
          ('AWEFUL', 'AWFUL'),
          ('AEA', '_EXCLUDE_'),
          ('ANWHERE', 'ANYWHERE'),
          ('APPLEHILL', 'APPLE HILL'),
          ('ARCHAEOLOGY', 'ARCHEOLOGY'),
          ('BARONIN', 'BARON IN'),
          ('BEGENT', 'REGENT'),
          ('BOUT', 'ABOUT'),
          ('BYE-BYE', 'BYE BYE'),
          ('BARNSWELL', 'BARNWELL'),
          ('BEDDINGTON', 'FEDDINGTON'),
          ('BUYBYE', 'BYE BYE'),
          ('BUH', 'BYE'),
          ('BUYBYE', 'BUY BYE'),
          ('BYEE', 'BYE'),
          ('CAMBRIGE', 'CAMBRIDGE'),
          ('CASTEL', 'CASTLE'),
          ('CASTAR', '_EXCLUDE_'),
          ('CAFFE', 'CAFE'),
          ('CASLE', 'CASTLE'),
          ('CASTE', 'CASTLE'),
          ('CASLTEHILL', 'CASTLEHILL'),
          ('CASTLE HILL', 'CASTLEHILL'),
          ('CASTLE HILL\'', 'CASTLEHILL\'S'),
          ('CATHERINE\'S\'S', 'CATHERINE\'S'),
          ('CCAN', 'CAN'),
          ('CCENTRAL', 'CENTRAL'),
          ('CENTRAP', 'CENTRAL'),
          ('CENTRE', 'CENTER'),
          ('CHEAPPRICERANGE', 'CHEAP PRICERANGE'),
          ('CHEAPRESTAURANT', 'CHEAP RESTAURANT'),
          ('CHEEP', 'CHEAP'),
          ('CHEAPRA', 'CHEAP'),
          ('CHEERY', 'CHERY'),
          ('CHERRY HINTON', 'CHERRYHINTON'),
          ('CHIDREN', 'CHILDREN'),
          ('CHINE', 'CHINESE'),
          ('CHINES', 'CHINESE'),
          ('CHINESE', 'CHINES'),
          ('CHLIDREN', 'CHILDREN'),
          ('CINESE', 'CHINESE'),
          ('COFF', 'COFFEE'),
          ('COFFE', 'COFFEE'),
          ('COFFEE', 'COFFE'),
          ('COFEE', 'COFFEE'),
          ('CONNCETION', 'CONNECTION'),
          ('CONECTION', 'CONNECTION'),
          ('CONNECTIO', 'CONNECTION'),
          ('CONTINTENAL', 'CONTINENTAL'),
          ('CONTEMPARY', 'CONTEMPORARY'),
          ('CONTEPORARY', 'CONTEMPORARY'),
          ('COUL', 'COULD'),
          ('DBAY', 'BAY'),
          ('DINTTON', 'DITTON'),
          ('DITION', 'DITTON'),
          ('DIRTON', 'DITTON'),
          ('DONT', 'DOESN\'T'),
          ('DON\'T\'', 'DOESN\'T'),
          ('DON;T', 'DOESN\'T'),
          ('DOES\'NT', 'DOESN\'T'),
          ('DOENS\'T', 'DOESN\'T'),
          ('DOESNT', 'DOESN\'T'),
          ('DOESRVE', 'DESERVE'),
          ('DOES\'T', 'DOESN\'T'),
          ('DONT\'T', 'DON\'T'),
          ('DONT\'', 'DON\'T'),
          ('DON`T', 'DON\'T'),
          ('DON’T', 'DON\'T'),
          ('IDON\'T', 'I DON\'T'),
          ('IJ', '_EXCLUDE_'),
          ('I;M', 'I\'M'),
          ('DOSEN\'T', 'DOESN\'T'),
          ('DOTHAT', 'DO THAT'),
          ('DUMBASS', 'DUMB ASS'),
          ('ENGINERING', 'ENGINEERING'),
          ('EDDINGBOROUGH', 'EDINBURGH'),
          ('EDINBOROUGH', 'EDINBURGH'),
          ('ENTERAINMENT', 'ENTERTAINMENT'),
          ('EXCELENT', 'EXCELLENT'),
          ('EXPANSIVE', 'EXPENSIVE'),
          ('EXPENCIVE', 'EXPENSIVE'),
          ('EXPENSIVEE', 'EXPENSIVE'),
          ('EXPENSIVEINDIAN', 'EXPENSIVE INDIAN'),
          ('FANDITTON', 'FENDITTON'),
          ('FANTASTICTHANK', 'FANTASTIC THANK'),
          ('FENDERTON', 'FENDITTON'),
          ('FENDISHON', 'FENDITTON'),
          ('FENITON', 'FENDITTON'),
          ('FENNITON', 'FENDITTON'),
          ('FINITON', 'FENDITTON'),
          ('FENDINGTON', 'FENDITTON'),
          ('FENDITON', 'FENDITTON'),
          ('FEN DITTON', 'FENDITTON'),
          ('FFSION', 'FUSION'),
          ('FAMILAR', 'FAMILIAR'),
          ('FINDA', 'FIND A'),
          ('FINDAMERICAN', 'FIND AMERICAN'),
          ('FO', 'FOR'),
          ('FOIR', 'FOR'),
          ('FOR-', 'FOR'),
          ('FODO', 'FOOD'),
          ('FORDABLE', 'AFORDABLE'),
          ('FUR', '_EXCLUDE_'),
          ('GALLERIA', 'GALLERY'),
          ('GERTEN', 'GIRTON'),
          ('GERTON', 'GIRTON'),
          ('GOOD0BYE', 'GOODBYE'),
          ('GODOBYE', 'GOODBYE'),
          ('GOOBYE', 'GOODBYE'),
          ('GOODBE', 'GOODBYE'),
          ('GOODBYW', 'GOODBYE'),
          ('GOODDBYE', 'GOODBYE'),
          ('GOODE', 'GOODBYE'),
          ('GOOFBYE', 'GOODBYE'),
          ('GOOODBYE', 'GOODBYE'),
          ('GOOD BYE', 'GOODBYE'),
          ('GOOD-BYE', 'GOODBYE'),
          ('GOODTHANK', 'GOOD THANK'),
          ('GOODWHAT', 'GOOD WHAT'),
          ('GOODYBE', 'GOODBYE'),
          ('GOODYE', 'GOODBYE'),
          ('GOO', 'GOOD'),
          ('GREATTHANK', 'GREAT THANK'),
          ('GREATWHAT', 'GRAT THANK'),
          ('GUESHOUSE', 'GUESTHOUSE'),
          ('HASTV', 'HAS TV'),
          ('HEDGES\'s', 'HEDGES'),
          ('HEDGERS', 'HEDGES'),
          ('HEGES', 'HEDGES'),
          ('HADGES', 'HEDGES'),
          ('HII', 'HILL'),
          ('HIL', 'HILL'),
          ('HINSON', 'HINSTON'),
          ('HITTON', 'HINSTON'),
          ('HTHE', '_EXCLUDE_'),
          ('IAM', 'I AM'),
          ('IAN', 'I AM'),
          ('II', 'I'),
          ('II\'M', 'I\'M'),
          ('I M', 'I AM'),
          ('I"M', 'I\'M'),
          ('IM', 'I\'M'),
          ('I\'\'M', 'I\'M'),
          ('INDIANINDIAN', 'INDIAN INDIAN'),
          ('INDITTON', 'IN DITTON'),
          ('INEXPRNSIVE', 'INEXPENSIVE'),
          ('INPRICERANGE', 'IN PRICERANGE'),
          ('INTERNATION', 'INTERNATIONAL'),
          ('INTERNNATIONAL', 'INTERNATIONAL'),
          ('INT HE', 'IN THE'),
          ('ISNT', 'ISN\'T'),
          ('JAP', '_EXCLUDE_'),
          ('\'KAY', 'KAY'),
          ('KINGS HEDGES', 'KINKGSHEDGES'),
          ('KINGTHE', 'KINGTHE'),
          ('KINKGSHEDGES', 'KINGSHEDGES'),
          ('LOOKINF', 'LOOKING'),
          ('LOOKIN', 'LOOKING'),
          ('MEDITERRARANEAN', 'MEDITERRANEAN'),
          ('MEXIAN', 'MEXICAN'),
          ('MIDDELE', 'MIDDLE'),
          ('MIDDLEEASTERN', 'MIDDLE EASTERN'),
          ('MODER', 'MODERN'),
          ('MOTAL', 'MOTEL'),
          ('MUCHHAVE', 'MUCH HAVE'),
          ('NEEDADDENBROOK\'S', 'NEED ADDENBROOK\'S'),
          ('NEEDA', 'NEED A'),
          ('NEEDEXPENSIVE', 'NEED EXPENSIVE'),
          ('NEEED', 'NEED'),
          ('NEWHAM', 'NEWNHAM'),
          ('NOCONTEMPORARY', 'NO CONTEMPORARY'),
          ('NODOES', 'NO DOES'),
          ('NOONAN', '_EXCLUDE_'),
          ('NUMBERAND', 'NUMBER AND'),
          ('NUMBERAND', 'NUMBERAND'),
          ('NUMMBER', 'NUMBER'),
          ('OFCOURSE', 'OF COURSE'),
          ('OKAY', 'OK'),
          ('OKDOES', 'OK DOES'),
          ('OKDO', 'OK DO'),
          ('OKEY', 'OK'),
          ('OKGOODBAY', 'OKGOODBAY'),
          ('OKHWAT', 'OK WHAT'),
          ('OKMAY', 'OK MAY'),
          ('OKTHANK', 'OK THANK'),
          ('OKWHAT\'S', 'OK WHAT\'S'),
          ('ON THE MODERATE', 'IN THE MODERATE'),
          ('OPENNING', 'OPENING'),
          ('OT', 'OR'),
          ('PHONBE', 'PHONE'),
          ('PHONEN', 'PHONE'),
          ('PHONENUMBER', 'PHONE NUMBER'),
          ('PHONME', 'PHONE'),
          ('PIRCE', 'PRICE'),
          ('PLACEWITH', 'PLACE WITH'),
          ('PLCE', 'PRICE'),
          ('PONE', 'PHONE'),
          ('POSTCODE', 'POST CODE'),
          ('PRCE', 'PRICE'),
          ('PRICEP', 'PRICE'),
          ('PRICERANGE', 'PRICE RANGE'),
          ('PRIVE', 'PRICE'),
          ('PRIZE', 'PRICE'),
          ('PSOT', 'POST'),
          ('PUBMODERATE', 'PUB MODERATE'),
          ('QUEENS\'', 'QUEEN\'S'),
          ('RANCHTHE', 'RANCH THE'),
          ('RAODSIDE', 'ROADSIDE'),
          ('RE', ''),
          ('REALLYUM', 'REALLY UM'),
          ('REASTURTANT', 'RESTAURANT'),
          ('REATAURANT', 'RESTAURANT'),
          ('REPET', 'REPEAT'),
          ('RESAURANT', 'RESTAURANT'),
          ('RESTAUANT', 'RESTAURANT'),
          ('RESTAURAN', 'RESTAURANT'),
          ('RESTAURANTE', 'RESTAURANT'),
          ('RESTAURANTI N', 'RESTAURANT IN'),
          ('RESTAURANTIN', 'RESTAURANT IN'),
          ('RESTAURAT', 'RESTAURANT'),
          ('RESTAUTANT', 'RESTAURANT'),
          ('RESTUARANT', 'RESTAURANT'),
          ('RESTURTANT', 'RESTAURANT'),
          ('RESTAURNAT', 'RESTAURANT'),
          ('RESTRAUNT', 'RESTAURANT'),
          ('RESTRAURANT', 'RESTAURANT'),
          ('RESTAURAUNT', 'RESTAURANT'),
          ('RESTAURANT\'S', 'RESTAURANTS'),
          ('RIVER SIDE', 'RIVERSIDE'),
          ('RIVESIDE', 'RIVERSIDE'),
          ('ROMSY', 'ROMSEY'),
          ('SHOUD', 'SHOULD'),
          ('SENDETON', 'FENDITTON'),
          ('SETTINGTON', '_EXCLUDE_'),
          ('SETTERTON', '_EXCLUDE_'),
          ('SHABENDA', '_EXCLUDE_'),
          ('SREVE', 'SERVE'),
          ('STROVER', '_EXCLUDE_'),
          ('TAKENMIX', '_EXCLUDE_'),
          ('SLI', '_EXCLUDE_'),
          ('SENDINGTON', 'FENDINGTON'),
          ('SENDITTON', 'FENDITTON'),
          ('SH', ''),
          ('SHAMPAIN', 'CHAMPAIN'),
          ('SHUSHI', 'SUSHI'),
          ('SILENCE', '(SIL)'),
          ('SILENT', '(SIL)'),
          ('SIL', '(SIL)'),
          ('SINDEENTAN', 'FENDITTON'),
          ('SINDEETAN', 'FENDITTON'),
          ('SINDINTON', 'FENDITTON'),
          ('SOMETHINGIN', 'SOMETHING IN'),
          ('SOMTHING', 'SOMETHING'),
          ('STAIONS', 'STATIONS'),
          ('STAION', 'STATION'),
          ('STANDARAD', 'STANDARD'),
          ('STREE', 'STREET'),
          ('ST', 'SAINT'),
          ('TEH', 'THE'),
          ('TELEVISON', 'TELEVISION'),
          ('TELEVSION', 'TELEVISION'),
          ('TELIVISION', 'TELEVISION'),
          ('TELIVISON', 'TELEVISION'),
          ('TEL', 'TELL'),
          ('THABK', 'THANK'),
          ('THAK', 'THANK'),
          ('THANH', 'THANK'),
          ('THA\'S', 'THAT\'S'),
          ('THATCHILDREN', 'THAT CHILDREN'),
          ('THATS', 'THAT'),
          ('THEADDRESS', 'THE ADDRESS'),
          ('THEBEST', 'THE BEST'),
          ('THEEXPENSIVE', 'THE EXPENSIVE'),
          ('THEFUSION', 'THE FUSION'),
          ('THEINTERVIEW', 'THE INTERVIEW'),
          ('THEIRE', 'THEIR'),
          ('THEPHONE', 'THE PHONE'),
          ('THEPRICERANGE', 'THE PRICERANGE'),
          ('THEPRICE', 'THE PRICE'),
          ('THEROMSEY', 'THE ROMSEY'),
          ('THEVENUE', 'THE VENUE'),
          ('THEY\'', 'THEY'),
          ('THNK', 'THANK'),
          ('TINKHAM', '_EXCLUDE_'),
          ('TNANK', 'THANK'),
          ('THNAK', 'THANK'),
          ('THER', '_EXCLUDE_'),
          ('THASNK', 'THANK\'S'),
          ('THANKTOU', 'THANK YOU'),
          ('TRUMPINGTONAREA', 'TRUMPINGTON AREA'),
          ('TRUMPINTON', 'TRUMPINGTON'),
          ('TRUMPTINGTON', 'TRUMPINGTON'),
          ('TRUNPINGTON', 'TRUMPINGTON'),
          ('TVE', '_EXCLUDE_'),
          ('TEX/MEX', 'TEXMEX'),
          ('THA', '_EXCLUDE_'),
          ('TRADIONAL', 'TRADITIONAL'),
          ('TRADITIONNAL', 'TRADITIONAL'),
          ('TRINTY', 'TRINITY'),
          ('TRUFFINTON', '_EXCLUDE_'),
          ('TUK', '_EXCLUDE_'),
          ('TOPPINTON', '_EXCLUDE_'),
          ('UNIVERCITY', 'UNIVERSITY'),
          ('VANUE', 'VENUE'),
          ('VENEUE', 'VENUE'),
          ('VENE', 'VENUE'),
          ('VODCA', 'VODKA'),
          ('WAHT', 'WHAT'),
          ('WANNT', 'WANT'),
          ('WANTA', 'WANT A'),
          ('WANTINTERNATIONAL', 'WANT INTERNATIONAL'),
          ('WEST SIDE', 'WESTSIDE'),
          ('WE\'', 'WE'),
          ('WHATADDRESS', 'WHAT ADDRESS'),
          ('WHATAREA', 'WHAT AREA'),
          ('WHATPRICE', 'WHAT PRICE'),
          ('WHATS', 'WHAT\'S'),
          ('WHATTHAT', 'WHAT THAT'),
          ('WHATTYPE', 'WHAT TYPE'),
          ('WHATWHAT', 'WHAT WHAT'),
          ('WELMBEY', '_EXCLUDE_'),
          ('WHNAT', 'WHAT'),
          ('WI-FI', 'WIFI'),
          ('WITHINTERNET', 'WITH INTERNET'),
          ('WITHWHAT', 'WITH WHAT'),
          ('VARNWELL', 'BARNWELL'),
          ('VEGATARIAN', 'VEGETARIAN'),
          ('VEGERTARIAN', 'VEGETARIAN'),
          ('WAHTS', 'WHAT\'S'),
          ('WOULDL', 'WOULD'),
          ('WOUD', 'WOULD'),
          ('WHT', 'WHY'),
          ('WAMSLEY', '_EXCLUDE_'),
          ('XPENSIVE', 'EXPENSIVE'),
          ('YEP', 'YUP'),
          ('YESI', 'YES'),
          ('YESYES', 'YES YES'),
          ('YOUTELL', 'YOU TELL'),
          ('YOUWHAT', 'YOU WHAT'),
          ('YOURE', 'YOUR'),
          ('YOU\'', 'YOU'),
          ('YUO', 'YOU'),
          ('YO', 'YOU'),
          ('ZIPCODE', 'ZIP CODE'),
          ('ZIZI', 'ZIZZI'),
          ('HGKBHBNKBN', '_EXCLUDE_'),
          ('GIRTIN', 'GIRTON'),
          ('GERTIN', 'GIRTON'),
          ('GRIFFON', '_EXCLUDE_'),
          ('GRITON', '_EXCLUDE_'),
          ('GURTIN', '_EXCLUDE_'),
          ('HADDIN', '_EXCLUDE_'),
          ('FOO', '_EXCLUDE_'),
          ('FRUMPTINGTON', '_EXCLUDE_'),
          ('FARMWELL', '_EXCLUDE_'),
          ('EPPING', '_EXCLUDE_'),
          ('ENTREES', '_EXCLUDE_'),
          ('ENTREE', '_EXCLUDE_'),
          ('DENTON', '_EXCLUDE_'),
          ('DERKIN', '_EXCLUDE_'),
          ('CURIOUSITY', 'CURIOSITY'),
          ('FYNE\'S', '_EXCLUDE_'),
          ('CITYTON', '_EXCLUDE_'),
          ('CINTINSIN', '_EXCLUDE_'),
          ('CINDOR', '_EXCLUDE_'),
          ('CINDINSIN', '_EXCLUDE_'),
          ('CINDINGTON', '_EXCLUDE_'),
          ('FORA', 'FOR A'),
          ('CITYCENTER', 'CITY CENTER'),
          ('CHESTERTOWN', 'CHESTERTON'),
          ('CHERRYHINT', 'CHERRYHINTON'),
          ('CHESTERON', 'CHESTERTON'),
          ('CHESTERSON', 'CHESTERTON'),
          ('CHESTERTIN', 'CHESTERTON'),
          ('CENTRY', 'CENTURY'),
          ('BENNINGHAM', '_EXCLUDE_'),
          ('BOUCHE', '_EXCLUDE_'),
          ('BUGGEN', '_EXCLUDE_'),
          ('CAMPTON', '_EXCLUDE_'),
          ('BBCAFE', '_EXCLUDE_'),
          ('WHEATSHEAF', '_EXCLUDE_'),
          ('WELMBEY', '_EXCLUDE_'),
          ('WAMSLEY', '_EXCLUDE_'),
          ('ANDERSBERG', '_EXCLUDE_'),
          ('ADDENBERG', '_EXCLUDE_'),
          ('ADDENBERGS', '_EXCLUDE_'),
          ('ADDINBERGS', '_EXCLUDE_'),
          ('FASTFOOD', 'FAST FOOD'),
          ('FEDDINGTON', '_EXCLUDE_'),
          ('FENN', '_EXCLUDE_'),
          ('HINSTON', '_EXCLUDE_'),
          ('HUFFINGTON', '_EXCLUDE_'),
          ('HUMBERSTONE', '_EXCLUDE_'),
          ('I\'', '_EXCLUDE_'),
          ('I\'S', '_EXCLUDE_'),
          ('HUNTINGDON', 'HUNTINGTON'),
          ('INTERANTIONAL', 'INTERNATIONAL'),
          ('INTERENT', 'INTERNET'),
          ('INTERNETIONAL', 'INTERNATIONAL'),
          ('I`D', 'I\'D'),
          ('I`M', 'I\'M'),
          ('I’M', 'I\'M'),
          ('JAPENSE', 'JAPANESE'),
          ('KINGSHEDGE', 'KINGSHEDGES'),
          ('LECTRON', '_EXCLUDE_'),
          ('LOKING', 'LOOKING'),
          ('MAYI', 'MAY I'),
          ('MEDERATELY', 'MODERATELY'),
          ('MEDIATRAIN', 'MEDITERRANEAN'),
          ('MEDITAREAN', 'MEDITERRANEAN'),
          ('MEDITARTIAN', 'MEDITERRANEAN'),
          ('MEDITATRIAN', 'MEDITERRANEAN'),
          ('MEDITERAINIAN', 'MEDITERRANEAN'),
          ('MEDITERANIAN', 'MEDITERRANEAN'),
          ('MEDITERRANIEN', 'MEDITERRANEAN'),
          ('MEDITERRANION', 'MEDITERRANEAN'),
          ('MEDITERREAN', 'MEDITERRANEAN'),
          ('MEDITTERRANEAN', 'MEDITERRANEAN'),
          ('MEXI CAN', 'MEXICAN'),
          ('MEXICAN/TEX', '_EXCLUDE_'),
          ('MH', '_EXCLUDE_'),
          ('MODERATLEY', 'MODERATELY'),
          ('MODERTLY', 'MODERATELY'),
          ('MORDERATELY', 'MODERATELY'),
          ('MOSERATELY', 'MODERATELY'),
          ('NMBER', 'NUMBER'),
          ('NOICE', 'NOISE'),
          ('NEWNHAMS', 'NEWNHAM'),
          ('NEWN', '_EXCLUDE_'),
          ('NEWCHESTERTEN', 'NEW CHESTERTON'),
          ('NEWCHESTERTON', 'NEW CHESTERTON'),
          ('NOOO', '_EXCLUDE_'),
          ('NORTHEN', 'NORTHERN'),
          ('OWULD', 'WOULD'),
          ('PENDINGTON', '_EXCLUDE_'),
          ('PHANE', '_EXCLUDE_'),
          ('PHOEN', 'PHONE'),
          ('PLEACE', 'PLEASE'),
          ('POSTEODE', 'POST CODE'),
          ('RANGW', 'RANGE'),
          ('REASTAURANT', 'RESTAURANT'),
          ('RESATAURANT', 'RESTAURANT'),
          ('RESTARANT', 'RESTAURANT'),
          ('RESTARAUBT', 'RESTAURANT'),
          ('RESTARAUNT', 'RESTAURANT'),
          ('RESTARUANT', 'RESTAURANT'),
          ('RESTAUARANT', 'RESTAURANT'),
          ('RESTAURANT;', 'RESTAURANT'),
          ('RESTAURATN', 'RESTAURANT'),
          ('RESTURANT', 'RESTAURANT'),
          ('RESTURANT', 'RESTAURANT'),
          ('ROSEY', '_EXCLUDE_'),
          ('SENDON', '_EXCLUDE_'),
          ('SORRENTO', '_EXCLUDE_'),
          ('SENDITON', 'FENDITTON'),
          ('THANKYOU', 'THANK YOU'),
          ('THANK\'S', 'THANKS'),
          ('TRUMPING', 'TRUMPINGTON'),
          ('UNINT', '_UNINT_'),
          ('YAY', '_EXCLUDE_'),
          ('YUH', '_EXCLUDE_'),
          ('YA', '_EXCLUDE_'),
          ('YEA', 'YEAH'),
          ('WULD', 'WOULD'),
          ('WHAT\'T', 'WHAT\'S'),
          ('WHAT;S', 'WHAT\'S'),
          ('WHAT`S', 'WHAT\'S'),
          ('WHAT´S', 'WHAT\'S'),
          ('WHAT’S', 'WHAT\'S'),
          ('THAT`S', 'THAT\'S'),
          ('THEE', 'THE'),
          ('ROSEY', 'ROMSEY'),
          ('SEVER', 'SERVE'),
          ('SEVERS', 'SERVES'),
          ('SITAR', '_EXCLUDE_'),
          ('MED', '_EXCLUDE_'),
          ('CUNT', '_EXCLUDE_'),
          ('MEXI', '_EXCLUDE_'),
          ('CHINES', 'CHINESE'),
          ('COFFE', 'COFFEE'),
          ('TO EA', 'TO EAT'),
          ('GE THE', 'GET THE'),
          ('TO IND', 'TO FIND'),
          ('RIVERSIDES AREA', 'RIVERSIDE AREA'),
          ('CB', '_EXCLUDE_'),
          ('JUROPA', 'EUROPE'),
          ('WHERES', '_EXCLUDE_'),
          ('HK', '_EXCLUDE_'),
          ('CAF', '_EXCLUDE_'),
           ]
#}}}
for idx, tup in enumerate(_subst):
    pat, sub = tup
    _subst[idx] = (re.compile(r'(^|\s){pat}($|\s)'.format(pat=pat)), ' '+sub+' ')

# hesitation expressions {{{
_hesitation = ['AAAA', 'AAA', 'AA', 'AAH', 'A-', "-AH-", "AH-", "AH.", "AH",
               "AHA", "AHH", "AHHH", "AHMA", "AHM", "ANH", "ARA", "-AR",
               "AR-", "-AR", "ARRH", "AW", "EA-", "-EAR", "-EECH", "\"EECH\"",
               "-EEP", "-E", "E-", "EH", "EM", "--", "ER", "ERM", "ERR",
               "ERRM", "EX-", "F-", "HM", "HMM", "HMMM", "-HO", "HUH", "HU",
               "HUM", "HUMM", "HUMN", "HUMN", "HUMPH", "HUP", "HUU", "-",
               "MM", "MMHMM", "MMM", "NAH", "OHH", "OH", "SH", "UHHH", "EMMM"
               "UHH", "UHM", "UH'", "UH", "UHUH", "UHUM", "UMH", "UMM", "UMN",
               "UM", "URM", "URUH", "UUH", "ARRH", "AW", "EM", "ERM", "ERR",
               "ERRM", "HUMN", "UM", "UMN", "URM", "AH", "ER", "ERM", "HUH",
               "HUMPH", "HUMN", "HUM", "HU", "SH", "UH", "UHUM", "UM", "UMH",
               "URUH", "MMMM", "MMM", "OHM", "UMMM", "MHMM", "EMPH", "HMPH",
               "UGH", "UHH", "UMMMMM", "SHH", "OOH", ]
# }}}
for idx, word in enumerate(_hesitation):
    _hesitation[idx] = re.compile(r'(^|\s){word}($|\s)'.format(word=word))

_more_spaces = re.compile(r'\s{2,}')
_sure_punct_rx = re.compile(r'[.?!",_]')
_parenthesized_rx = re.compile(r'\(+([^)]*)\)+')


def normalise_text(text):
    """
    Normalises the transcription.  This is the main function of this module.
    """
    text = _sure_punct_rx.sub(' ', text)
    text = text.strip().upper()

    # Do dictionary substitutions.
    for pat, sub in _subst:
        text = pat.sub(sub, text)
    for word in _hesitation:
        text = word.sub(' (HESITATION) ', text)
    text = _more_spaces.sub(' ', text).strip()
    
    # Handle non-speech events (separate them from words they might be
    # agglutinated to, remove doubled parentheses, and substitute the known
    # non-speech events with the forms with underscores).
    #
    # This step can incur superfluous whitespace.
    if '(' in text:
        text = _parenthesized_rx.sub(r' (\1) ', text)
        for parenized, uscored in _nonspeech_trl.iteritems():
            text = text.replace(parenized, uscored)
        text = _more_spaces.sub(' ', text.strip())

    # remove duplicate non-speech events
    for pat, sub in _nonspeech_events:
        text = pat.sub(sub, text)
    text = _more_spaces.sub(' ', text).strip()

    for char in '^':
        text = text.replace(char, '')

    return text

_excluded_characters = ['=', '-', '*', '+', '~', '(', ')', '[', ']', '{', '}', '<', '>',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def exclude_asr(text):
    """
    This function is used for determining whether the transcription can be used for training ASR.

    Determines whether `text' is not good enough and should be excluded.
    "Good enough" is defined as containing none of `_excluded_characters' and being
    longer than one word.
    """
    if '_EXCLUDE_' in text:
        return True

    if text in ['_SIL_', ]:
        return True

    if text in ['_NOISE_', '_EHM_HMM_', '_INHALE_', '_LAUGH_']:
        return False

    # allow for sentences with these non-speech events if mixed with text
    for s in ['_NOISE_', '_INHALE_', '_LAUGH_']:
        text = text.replace(s,'')

    for char in _excluded_characters:
        if char in text:
            return True
    if '_' in text:
        return True

    if len(text) < 2:
        return True

    return False

def exclude_lm(text):
    """
    This function is used for determining whether the transcription can be used for Language Modeling.

    Determines whether `text' is not good enough and should be excluded.
    "Good enough" is defined as containing none of `_excluded_characters' and being
    longer than one word.
    """

    if text.find('_EXCLUDE_') >= 0:
        return True

    for char in _excluded_characters:
        if char in text:
            return True

    return False

def exclude_slu(text):
    """
    This function is used for determining whether the transcription can be used for training Spoken Language Understanding.
    """
    return exclude_lm(text)

def exclude_by_dict(text, known_words):
    """
    Determines whether text is not good enough and should be excluded.

    "Good enough" is defined as having all its words present in the
    `known_words' collection."""
    return not all(map(lambda word: word in known_words, text.split()))
