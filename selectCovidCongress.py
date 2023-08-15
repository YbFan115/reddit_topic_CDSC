#!/usr/bin/env python3

###############################################################################
##
## The purpose of this script is to filter through congressional speech and limit
## it to covid-related and coronavirus-related speech
##
## 1 - Process args
##
## 2 - Read through each JSON file in turn
##
## 3 - Report results
##
###############################################################################
##
## Parsed Congressional Record Files
##
## {"header": {...}, "content": [{"kind": "speech", "speaker": "...", "text": "" }]}
## 
## this json file is a list of items, and the bits we want are 'kind, speaker, text, ....' inside {}
##
## Members of Congress Dataset
##
## Two files:
## legislators-current.csv  
## legislators-historical.csv 
##
## Format:
## last_name,first_name,middle_name,suffix,nickname,full_name,birthday,gender,type,state,district,senate_class,party,url,address,phone,contact_form,rss_url,twitter,facebook,youtube,youtube_id,bioguide_id,thomas_id, opensecrets_id,lis_id,fec_ids,cspan_id,govtrack_id,votesmart_id,ballotpedia_id,washington_post_id,icpsr_id,wikipedia_id
##

import json
from os import listdir
from os.path import isfile, join
import glob
import re
import pandas as pd
import numpy as np
import us


## prod file
searchString = "2017/" + "*/json/*.json"
outfile = 'gunUtterances.tsv'
#ambigfile = 'ambigfile.tsv'
#searchString = "2020/" + "CREC-2020-03*/json/*.json"
#outfile = '20200301covidUtterances.tsv'
#ambigfile = '20200301ambigfile.tsv'

## test with just 2020-03-25
#searchString = "2020/" + "CREC-2020-03-25/json/*.json"
#outfile = 'test_outfile.tsv'
#ambigfile = 'test_ambigfile.tsv'

#print(searchString)
fileList = glob.glob(searchString)
outFH = open(outfile, 'w')
outFH.write(f"speechdate\tspeakername\tis_republican\tspeech\n")
#ambigFH = open(ambigfile, 'w')
#ambigFH.write(f"speechdate\tspeakername\tparty\tspeech\n")
gun_re = re.compile(r'(gun)', flags=re.IGNORECASE)

#read in list of representatives and senators...
partyDF = pd.read_csv('legislators-current.csv')
fixPartyDF = pd.read_csv('cleanMappings.tsv', sep='\t')
#print(fixPartyDF)

# make sure that last name (which is what the CR uses...) is actually a unique key
dupDF = partyDF[partyDF.duplicated(['last_name'], keep=False)]
dupDF = dupDF.sort_values(by='last_name')
dupDF.to_csv('duplicates.tsv', sep='\t', index_label=False)
#print(f"Keep an eye on duplicates....{dupDF}")

partyDF['caps_name'] = partyDF['last_name'].apply(lambda x:x.upper()) 
partyDF['state_name'] = partyDF['state'].apply(lambda x:us.states.lookup(str(x))) ##this works
partyDF['caps_state_name'] = partyDF['caps_name'] + " OF " + partyDF['state_name'].apply(lambda x:str(x).upper())
#us.states.lookup returns state name from postal abbreviation
partyDF.to_csv('partyDataExpanded.tsv', sep='\t') #save out the dataset we used

for filename in fileList: 
	with open(filename) as fh: 
		jData = json.load(fh) 
		byUtterance = jData["content"] 
		for u in byUtterance: 
			if (u["kind"] != "speech"):
                		continue 
			gunMatches = gun_re.search(u["text"]) 
			if (gunMatches): #write the speech to our record 
				row = ''
				if ("PRESIDING" in u["speaker"]):
					#print(f"Found a match to ignore: {u}")
					continue
				if ("SPEAKER" in u["speaker"]):
				#if (str(u["speaker"].split(' ')[0]) == "PRESIDING"):
					#print(f"Found a match to ignore: {u}")
					continue
				#print(f"Found a match: {u}")
				speechdate = filename.split('/')[-1] 
				speechdate = speechdate[:-5] 
				speakername = u["speaker"].upper()  #that's how it is in the CR sadly
				speakername = speakername.split(' ')[1:] #no Mr. or Ms. etc.
				speakername = str(speakername).lstrip('[').rstrip(']')
				speakername = speakername.replace("'", '')
				speakername = speakername.replace(",", '')
				speakername = speakername.replace('"', '')
				#df[df['model'].str.match('Mac')]
				littlerow = fixPartyDF[fixPartyDF['rendered_name'].str.fullmatch(speakername)] #check exceptions first; must be exact match
				if not (littlerow.empty): #match in exception list
					partyAff = littlerow.get('party').values
					partyAff = str(partyAff).lstrip('[').rstrip(']')
					partyAff = partyAff.replace("'", '')
					partyAff = partyAff.replace(",", '')
					if (" " in partyAff):
						print(f"Found multiple entries in {partyAff} for {speakername}")
						print(littlerow)
				else:
					row = partyDF.loc[partyDF.caps_name == speakername]
					partyAff = row.get('party').values
					partyAff = str(partyAff).lstrip('[').rstrip(']')
					partyAff = partyAff.replace("'", '')
					partyAff = partyAff.replace(",", '')
					#print(f"then got to here -- {partyAff}") #OK
					partyAff = str(partyAff)
					if (partyAff == ''): #zero or multiple with that name
						#print(f"len fail here and got to here -- going to try caps name {partyAff}")
						row = partyDF.loc[partyDF.caps_state_name == speakername] #try again with state tag
						partyAff = row.get('party').values
						partyAff = str(partyAff).lstrip('[').rstrip(']')
						partyAff = partyAff.replace("'", '')
						partyAff = partyAff.replace(",", '')
						#print(f"...and got to here -- {partyAff}")
						partyAff = str(partyAff)
						if (partyAff == ''): #zero or multiple with that name
							print(f"still no clear match for {speakername}")
							#print(f"and got to here -- {partyAff}")
							partyAff = None
				speech = u["text"] 
				speech = speech.replace('\n', '') #verrrrry long lines :/ will this work?
				#often the parsed speeches look like this: "     Ms. PELOSI.  <greeting or direction of comment> <actual text here>" -- I want to strip the attribution but not to whom it is directed
				speech = speech.lstrip() #remove all the leading whitespace
				speech = re.sub(r'^Mr\.', '', speech) #remove leading titles
				speech = re.sub(r'^Ms\.', '', speech) #remove leading titles
				speech = re.sub(r'^Mrs\.', '', speech) #remove leading titles
				speech = speech.lstrip()
				speech = re.sub(speakername, '', speech) 
				speech = speech.lstrip('.') 
				speech = speech.lstrip()
				if (str(partyAff) == 'Independent'):
					continue
				elif (str(partyAff) == 'Libertarian'):
					continue
				elif (str(partyAff) == 'Republican'):
					partyAff = 1
				elif (str(partyAff) == 'Democrat'):
					partyAff = 0
				else:
					print(f"Multi-party affiliation found {partyAff} for {speakername} -- it was {u}")
					continue

				outFH.write(f"{speechdate}\t{speakername}\t{str(partyAff)}\t{speech}\n")
				#print('On to the next')

