Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=d9fa583a4aa9dafe ===
Gold sentences and facts:
Sentence 0: No. in series No. in season Title Directed by Written by Original air date Production code U.S. viewers ( millions ) 156 `` Playing Dead '' Michael Smith Antoinette Stella April 19 , 2009 ( 2009 - 04 - 19 ) ( USA ) May 6 , 2009 ( 2009 - 05 - 06 ) ( NBC ) 08009 4.57 Detectives Goren and Eames investigate the shooting of Stacy Hayes - Fitzgerald and her boyfriend Rick Siebert , two drug addicts who are blackmailing Stacy 's stepfather , Neil Hayes - Fitzgerald ( Scott Cohen ) , a politician running for mayor . Rick was killed in the shooting , but Stacy survived by playing dead for hours under his bleeding corpse until they were found by Stacy 's mother , Josie . After returning home from the hospital , Stacy attempts to commit suicide . After seeing how she reacts when Neil is around , Goren asks Stacy if she has been abused , but she denies it . Goren discovers that Stacy and her family moved to Switzerland for a year when she was in high school , and realizes that Stacy 's younger sister Sophie is actually her daughter by Neil . While questioning Neil and his mother , Camille ( Kathy Baker ) , it is revealed that she had Rick killed because he was threatening to expose Neil 's family scandal , thwarting his mayoral bid . 157 `` Rock Star '' Bill D'Elia Ed Zuckerman April 26 , 2009 ( 2009 - 04 - 26 ) ( USA ) May 13 , 2009 ( 2009 - 05 - 13 ) ( NBC ) 08008 4.10 Detective Wheeler and her new partner , Detective Zack Nichols , investigate the stabbing of an artist , who was found in a neighborhood rife with ethnic tensions . 158 `` Identity Crisis '' Michael Smith Pamela Wechsler May 3 , 2009 ( 2009 - 05 - 03 ) ( USA ) May 20 , 2009 ( 2009 - 05 - 20 ) ( NBC ) 08005 3.14 Two young brothers live with their schizophrenic mother , who dies in the bathtub by electrocution . They are questioned about the turn of events , and the younger brother Anthony tells police that he saw his brother Tommy push the electric heater into the tub . The two are subsequently separated and put into foster care . Twenty years later , Anthony reappears in Tommy 's life , and Tommy responds by shooting him . Goren and Eames are brought in to investigate the murder , and Rodgers tells them that the victim , Anthony , was dying from lymphoma . Meanwhile , over the years Tommy ( Sam Trammell ) has been living a mysterious life , pretending that he is a graduate of Princeton University and taking on false identities . He rents a house on Nantucket under the name Tyler Chisholm , and the owner 's daughter takes a liking to him . The owner is suspicious and runs a background credit check on `` Tyler . '' The credit check is reported to the detectives , as Tyler Chisholm was one of the names noted when interviewing people at the Princeton Eating Club . They travel to Nantucket , where they find Tommy and take him into custody . Back in New York , Eames questions the owner 's daughter and asks if there was anything strange or out of the ordinary that she could recall . She tells Eames that she remembers Tommy throwing something into the water and staring at it . Goren then questions Tommy and reveals that Anthony did not return to bring back old memories and haunt Tommy , but to make amends because of his imminent death . 159 `` In Treatment '' Jean de Segonzac Timothy J. Lea May 10 , 2009 ( 2009 - 05 - 10 ) ( USA ) May 27 , 2009 ( 2009 - 05 - 27 ) ( NBC ) 08010 3.14 Detectives Nichols and Wheeler investigate the murder of a Wall Street executive who is murdered during a gala . Nichols believes that his company is more involved with the case than they have admitted to the police . During the investigation the detectives discover that news of a falling stock , which plunged more after news of a fraud investigation , is tied to this murder . However , their investigation leads them to a treatment clinic , where their prime suspect is under the care of a controlling psychiatrist ( Dennis Boutsikaris ) . 160 5 `` Faithfully '' Jean de Segonzac Antoinette Stella May 17 , 2009 ( 2009 - 05 - 17 ) ( USA ) June 29 , 2009 ( 2009 - 06 - 29 ) ( NBC ) 08001 3.70 When a devout celebrity doctor is murdered , Goren and Eames enter the unexpectedly steamy world behind the pulpit . 161 6 `` Astoria Helen '' Norberto Barba Timothy J. Lea May 31 , 2009 ( 2009 - 05 - 31 ) ( USA ) July 6 , 2009 ( 2009 - 07 - 06 ) ( NBC ) 08004 3.97 A charming con man named Joe Gallagher ( Creighton James ) targets a lonely , single mother named Helen Bramer ( Arija Bareikis ) to try to hit an armored truck carrying millions of dollars , Bramer worked for the company of the armored truck . Detectives Nichols and Wheeler investigate when one of the con man 's partners dies weeks later in a bomb blast . Bramer and Gallagher become the prime suspects in the case when a van owner named Frank Stroup ( Domenick Lombardozzi ) is ripped off and later shot . But the detectives later learn that Bramer and Gallagher are n't the prime suspects when Bramer 's son Kevin gets abducted by Frank Stroup , and to rescue the boy , Detective Nichols has to put himself in the middle of a hostage situation . 162 7 `` Folie à Deux '' David Manson Michael S. Chernuchin June 7 , 2009 ( 2009 - 06 - 07 ) ( USA ) July 13 , 2009 ( 2009 - 07 - 13 ) ( NBC ) 08003 4.01 Detectives Goren and Eames investigate when a couple 's 2 - year - old daughter mysteriously disappears from her room during a robbery at the hotel , where her parents dined . They learn that a family member ( Lynn Redgrave ) is financially supporting the couple , while also dying of a heart condition . Glass fragments and a stolen key - card become substantial evidence which leads them to the robber , although there is no child with him . Further investigation leads Goren and Eames to receive a ransom call . However , they learn that someone relatively close to the family was taking advantage of the crime , attempting to win money over in the process . The detectives find out the parents have been giving a false description . There never was a child at the hotel . As they uncover a shocking conspiracy , Goren and Eames learn that the unstable mother ( Piper Perabo ) has been delusional about the months - ago accidental death of her daughter , prompting her husband ( Luke Kirby ) to exploit her condition in order to continue receiving financial support from his rich but childless aunt . Partially inspired by the Madeleine McCann and Caylee Anthony cases . 163 8 `` The Glory That Was ... '' Norberto Barba Robert Nathan June 14 , 2009 ( 2009 - 06 - 14 ) ( USA ) July 20 , 2009 ( 2009 - 07 - 20 ) ( NBC ) 08002 4.14 What should have been a simple meeting turns to murder , leaving Detectives Wheeler and Nichols to question an embassy , a security company , the victim 's relatives , and how it all connects to the Olympic Site Selection Committee . This episode is omitted in the DVD release of season 8 . 164 9 `` Family Values '' Jean de Segonzac Teleplay by : Walon Green Story by : Walon Green & Antoinette Stella June 21 , 2009 ( 2009 - 06 - 21 ) ( USA ) August 3 , 2009 ( 2009 - 08 - 03 ) ( NBC ) 08015 3.44 When a flier from a local Catholic high school drama production connects three murders , Detectives Goren and Eames are back on the case to understand the psychology of a religious fanatic before he claims another victim . 165 10 `` Salome in Manhattan '' `` Salome '' Steve Shill Andrew Lipsitz June 28 , 2009 ( 2009 - 06 - 28 ) ( USA ) August 10 , 2009 ( 2009 - 08 - 10 ) ( NBC ) 08006 N / A A celebrity chef 's star is on the rise as he plans the opening of a trendy restaurant with powerful partners . Yet his world is upended when his tabloid - friendly , gorgeous girlfriend and business partner is killed under mysterious circumstances . Detectives Nichols and Wheeler enter the seductive world of power and privilege and discover desperation and murder , but the more they learn about the victim and the men in her life , the more they learn about who the true killer is . 166 11 `` Lady 's Man '' Ken Girotti Michael S. Chernuchin June 28 , 2009 ( 2009 - 06 - 28 ) ( USA ) August 17 , 2009 ( 2009 - 08 - 17 ) ( NBC ) 08011 N / A Detectives Goren and Eames are back on the case to investigate the killing of a man who became famous for dodging a murder conviction because of a mishandled trial . The man spun his notoriety into a TV show but turned up dead on his birthday . Eames had worked the case of his first wife 's murder 10 years earlier . The case also involves an ADA ( Raul Esparza ) who has proven a thorn in Eames ' side more than once . 167 12 `` Passion '' Jonathan Herron Michael S. Chernuchin July 12 , 2009 ( 2009 - 07 - 12 ) ( USA ) August 24 , 2009 ( 2009 - 08 - 24 ) ( NBC ) 08012 3.47 Egocentric poet Jacob Garrety ( Will Chase ) would do anything to keep his poetry journal , `` The Village Quarterly '' , afloat . When foundation president Don McCallum shows up and announces he is cutting funding to the journal , Jacob appeals to assistant and lover Lauren to use her sexuality to change Don 's mind . Though reluctant , she agrees . While Jacob enjoys the advances of arts patron Sandra Dunbar ( Sarah Rafferty ) at a poetry reading , Lauren carries out Jacob 's wish , and winds up dead on the street . Detectives Nichols and Wheeler discover the deadly passion that exists behind the arts . 168 13 `` All In '' David Manson Teleplay by : Pamela Wechsler , Antoinette Stella , & Walon Green Story by : Pamela Wechsler July 19 , 2009 ( 2009 - 07 - 19 ) ( USA ) August 31 , 2009 ( 2009 - 08 - 31 ) ( NBC ) 08013 4.14 When poker genius Josh Snow ( Aaron Stanford ) loses $80,000 in a card game , backer and bookie Lou Cardinale ( Boris McGiver ) forces him into collections to work off the losses . Lou gives him a . 38 and a box of blanks to extract gambling debts , and sends along his mistress , Angela ( Aleksa Palladino ) , to keep an eye on the money . When Josh first collects from a guy in the local neighborhood , he shoots a blank at him . The guy gave him twenty thousand dollars and then he collects from Kip McGonagle ( Robert Leeshock ) . He fires the gun expecting a blank , but a real bullet kills Kip in the street . Detectives Goren and Eames , who know Snow from a former case ( identical historical plot elements , but different actor and character 's name , from season five 's `` Cruise to Nowhere '' ) , enter an elaborate game where they must join in the play , or be played . 169 14 `` Major Case '' Chris Zalla Andrew Lipsitz July 26 , 2009 ( 2009 - 07 - 26 ) ( USA ) August 31 , 2009 ( 2009 - 08 - 31 ) ( NBC ) 08014 4.63 A drug dealer is killed the night before she plans to leave town to live on her cousin 's farm . After Wheeler goes into labor , Eames pairs up with Nichols to investigate and find the girl 's killer . But when Nichols suspects the girl 's killer is an old friend ( Dylan Baker ) who works in the medical examiner 's office , Nichols does all he can to prove that he killed the victim , but it could cost Nichols his job in the process . Last episode to feature Julianne Nicholson ( Detective Megan Wheeler ) ; she is replaced by Kathryn Erbe ( Detective Alexandra Eames ) for the remainder of season eight . As of season nine , she is replaced by Saffron Burrows ( Detective Serena Stevens ) . 170 15 `` Alpha Dog '' Norberto Barba Walon Green August 2 , 2009 ( 2009 - 08 - 02 ) ( USA ) September 7 , 2009 ( 2009 - 09 - 07 ) ( NBC ) 08007 4.23 A hunky posterboy with a million - dollar torso has what others crave : sex appeal . When he shows up dead after a night of sex and drugs with an elusive woman , detectives Goren and Eames uncover the sexual paranoia behind the murder . 171 16 `` Revolution '' John David Coles Michael S. Chernuchin August 9 , 2009 ( 2009 - 08 - 09 ) ( USA ) September 7 , 2009 ( 2009 - 09 - 07 ) ( NBC ) 08016 4.83 A revolutionary plans the kidnapping of a powerful banking executive , but the heist goes wrong and the executive is killed . While probing the murder , Nichols and Eames must also prevent a terrorist campaign . After this episode , Ed Zuckerman and Michael S. Chernuchin leave the show as executive producers . Zuckerman returned to Law & Order as its executive producer and Chernuchin as its co-executive producer . Walon Green takes over as executive producer / show runner for all episodes .
  Fact 0.0: Episode 156, 'Playing Dead', was directed by Michael Smith.
  Fact 0.1: Episode 156, 'Playing Dead', was written by Antoinette Stella.
  Fact 0.2: Episode 156, 'Playing Dead', first aired in the USA on April 19, 2009.
  Fact 0.3: Episode 156, 'Playing Dead', aired on NBC on May 6, 2009.
  Fact 0.4: Episode 156, 'Playing Dead', has production code 08009.
  Fact 0.5: Episode 156, 'Playing Dead', had 4.57 million U.S. viewers.
  Fact 0.6: Detectives Goren and Eames investigate the shooting of Stacy Hayes-Fitzgerald and her boyfriend Rick Siebert.
  Fact 0.7: Stacy Hayes-Fitzgerald and Rick Siebert are two drug addicts who are blackmailing Stacy's stepfather, Neil Hayes-Fitzgerald.
  Fact 0.8: Neil Hayes-Fitzgerald, played by Scott Cohen, is a politician running for mayor.
  Fact 0.9: Rick was killed in the shooting.
  Fact 0.10: Stacy survived the shooting by playing dead for hours under Rick's bleeding corpse.
  Fact 0.11: Stacy was found by her mother, Josie.
  Fact 0.12: After returning home from the hospital, Stacy attempts to commit suicide.
  Fact 0.13: After seeing how Stacy reacts when Neil is around, Goren asks Stacy if she has been abused.
  Fact 0.14: Stacy denies being abused.
  Fact 0.15: Goren discovers that Stacy and her family moved to Switzerland for a year when Stacy was in high school.
  Fact 0.16: Goren realizes that Stacy's younger sister Sophie is actually Stacy's daughter by Neil.
  Fact 0.17: While questioning Neil and his mother, Camille (played by Kathy Baker), it is revealed that Camille had Rick killed.
  Fact 0.18: Camille had Rick killed because Rick was threatening to expose Neil's family scandal, which would thwart Neil's mayoral bid.
  Fact 0.19: Episode 157, 'Rock Star', was directed by Bill D'Elia.
  Fact 0.20: Episode 157, 'Rock Star', was written by Ed Zuckerman.
  Fact 0.21: Episode 157, 'Rock Star', first aired in the USA on April 26, 2009.
  Fact 0.22: Episode 157, 'Rock Star', aired on NBC on May 13, 2009.
  Fact 0.23: Episode 157, 'Rock Star', has production code 08008.
  Fact 0.24: Episode 157, 'Rock Star', had 4.10 million U.S. viewers.
  Fact 0.25: Detective Wheeler and her new partner, Detective Zack Nichols, investigate the stabbing of an artist.
  Fact 0.26: The artist was found in a neighborhood rife with ethnic tensions.
  Fact 0.27: Episode 158, 'Identity Crisis', was directed by Michael Smith.
  Fact 0.28: Episode 158, 'Identity Crisis', was written by Pamela Wechsler.
  Fact 0.29: Episode 158, 'Identity Crisis', first aired in the USA on May 3, 2009.
  Fact 0.30: Episode 158, 'Identity Crisis', aired on NBC on May 20, 2009.
  Fact 0.31: Episode 158, 'Identity Crisis', has production code 08005.
  Fact 0.32: Episode 158, 'Identity Crisis', had 3.14 million U.S. viewers.
  Fact 0.33: Two young brothers live with their schizophrenic mother, who dies in the bathtub by electrocution.
  Fact 0.34: The brothers are questioned about the turn of events.
  Fact 0.35: The younger brother, Anthony, tells police that he saw his brother Tommy push the electric heater into the tub.
  Fact 0.36: The two brothers are separated and put into foster care.
  Fact 0.37: Twenty years later, Anthony reappears in Tommy's life.
  Fact 0.38: Tommy responds by shooting Anthony.
  Fact 0.39: Detectives Goren and Eames are brought in to investigate the murder.
  Fact 0.40: Rodgers tells Goren and Eames that the victim, Anthony, was dying from lymphoma.
  Fact 0.41: Over the years, Tommy (played by Sam Trammell) has been living a mysterious life.
  Fact 0.42: Tommy pretends that he is a graduate of Princeton University and takes on false identities.
  Fact 0.43: Tommy rents a house on Nantucket under the name Tyler Chisholm.
  Fact 0.44: The owner's daughter takes a liking to Tommy.
  Fact 0.45: The owner is suspicious and runs a background credit check on 'Tyler'.
  Fact 0.46: The credit check is reported to the detectives.
  Fact 0.47: Tyler Chisholm was one of the names noted when interviewing people at the Princeton Eating Club.
  Fact 0.48: The detectives travel to Nantucket, where they find Tommy and take him into custody.
  Fact 0.49: Back in New York, Eames questions the owner's daughter and asks if there was anything strange or out of the ordinary that she could recall.
  Fact 0.50: The owner's daughter tells Eames that she remembers Tommy throwing something into the water and staring at it.
  Fact 0.51: Goren questions Tommy and reveals that Anthony did not return to bring back old memories and haunt Tommy.
  Fact 0.52: Goren reveals that Anthony returned to make amends because of Anthony's imminent death.
  Fact 0.53: Episode 159, 'In Treatment', was directed by Jean de Segonzac.
  Fact 0.54: Episode 159, 'In Treatment', was written by Timothy J. Lea.
  Fact 0.55: Episode 159, 'In Treatment', first aired in the USA on May 10, 2009.
  Fact 0.56: Episode 159, 'In Treatment', aired on NBC on May 27, 2009.
  Fact 0.57: Episode 159, 'In Treatment', has production code 08010.
  Fact 0.58: Episode 159, 'In Treatment', had 3.14 million U.S. viewers.
  Fact 0.59: Detectives Nichols and Wheeler investigate the murder of a Wall Street executive who is murdered during a gala.
  Fact 0.60: Nichols believes that the Wall Street executive's company is more involved with the case than the company has admitted to the police.
  Fact 0.61: During the investigation, the detectives discover that news of a falling stock is tied to this murder.
  Fact 0.62: The falling stock plunged more after news of a fraud investigation.
  Fact 0.63: The investigation leads the detectives to a treatment clinic.
  Fact 0.64: Their prime suspect is under the care of a controlling psychiatrist (played by Dennis Boutsikaris).
  Fact 0.65: Episode 160, 'Faithfully', is season 5 episode 5.
  Fact 0.66: Episode 160, 'Faithfully', was directed by Jean de Segonzac.
  Fact 0.67: Episode 160, 'Faithfully', was written by Antoinette Stella.
  Fact 0.68: Episode 160, 'Faithfully', first aired in the USA on May 17, 2009.
  Fact 0.69: Episode 160, 'Faithfully', aired on NBC on June 29, 2009.
  Fact 0.70: Episode 160, 'Faithfully', has production code 08001.
  Fact 0.71: Episode 160, 'Faithfully', had 3.70 million U.S. viewers.
  Fact 0.72: A devout celebrity doctor is murdered.
  Fact 0.73: Goren and Eames enter the unexpectedly steamy world behind the pulpit.
  Fact 0.74: Episode 161, 'Astoria Helen', is season 6 episode 6.
  Fact 0.75: Episode 161, 'Astoria Helen', was directed by Norberto Barba.
  Fact 0.76: Episode 161, 'Astoria Helen', was written by Timothy J. Lea.
  Fact 0.77: Episode 161, 'Astoria Helen', first aired in the USA on May 31, 2009.
  Fact 0.78: Episode 161, 'Astoria Helen', aired on NBC on July 6, 2009.
  Fact 0.79: Episode 161, 'Astoria Helen', has production code 08004.
  Fact 0.80: Episode 161, 'Astoria Helen', had 3.97 million U.S. viewers.
  Fact 0.81: A charming con man named Joe Gallagher (played by Creighton James) targets a lonely, single mother named Helen Bramer (played by Arija Bareikis).
  Fact 0.82: Joe Gallagher targets Helen Bramer to try to hit an armored truck carrying millions of dollars.
  Fact 0.83: Helen Bramer worked for the company of the armored truck.
  Fact 0.84: Detectives Nichols and Wheeler investigate when one of the con man's partners dies weeks later in a bomb blast.
  Fact 0.85: Helen Bramer and Joe Gallagher become the prime suspects when a van owner named Frank Stroup (played by Domenick Lombardozzi) is ripped off and later shot.
  Fact 0.86: The detectives later learn that Helen Bramer and Joe Gallagher are not the prime suspects when Helen Bramer's son Kevin gets abducted by Frank Stroup.
  Fact 0.87: To rescue the boy, Detective Nichols has to put himself in the middle of a hostage situation.
  Fact 0.88: Episode 162, 'Folie a Deux', is season 7 episode 7.
  Fact 0.89: Episode 162, 'Folie a Deux', was directed by David Manson.
  Fact 0.90: Episode 162, 'Folie a Deux', was written by Michael S. Chernuchin.
  Fact 0.91: Episode 162, 'Folie a Deux', first aired in the USA on June 7, 2009.
  Fact 0.92: Episode 162, 'Folie a Deux', aired on NBC on July 13, 2009.
  Fact 0.93: Episode 162, 'Folie a Deux', has production code 08003.
  Fact 0.94: Episode 162, 'Folie a Deux', had 4.01 million U.S. viewers.
  Fact 0.95: Detectives Goren and Eames investigate when a couple's 2-year-old daughter mysteriously disappears from her room during a robbery at the hotel where her parents dined.
  Fact 0.96: The couple is financially supported by a family member (played by Lynn Redgrave), who is also dying of a heart condition.
  Fact 0.97: Glass fragments and a stolen key-card become substantial evidence that leads the detectives to the robber, although there is no child with him.
  Fact 0.98: Goren and Eames receive a ransom call.
  Fact 0.99: They learn that someone relatively close to the family was taking advantage of the crime, attempting to win money in the process.
  Fact 0.100: The detectives find out the parents have been giving a false description.
  Fact 0.101: There never was a child at the hotel.
  Fact 0.102: The unstable mother (played by Piper Perabo) has been delusional about the months-ago accidental death of her daughter.
  Fact 0.103: The husband (played by Luke Kirby) exploits his wife's condition to continue receiving financial support from his rich but childless aunt.
  Fact 0.104: The episode is partially inspired by the Madeleine McCann and Caylee Anthony cases.
  Fact 0.105: Episode 163, 'The Glory That Was ...', is season 8 episode 8.
  Fact 0.106: Episode 163, 'The Glory That Was ...', was directed by Norberto Barba.
  Fact 0.107: Episode 163, 'The Glory That Was ...', was written by Robert Nathan.
  Fact 0.108: Episode 163, 'The Glory That Was ...', first aired in the USA on June 14, 2009.
  Fact 0.109: Episode 163, 'The Glory That Was ...', aired on NBC on July 20, 2009.
  Fact 0.110: Episode 163, 'The Glory That Was ...', has production code 08002.
  Fact 0.111: Episode 163, 'The Glory That Was ...', had 4.14 million U.S. viewers.
  Fact 0.112: What should have been a simple meeting turns to murder.
  Fact 0.113: Detectives Wheeler and Nichols question an embassy, a security company, the victim's relatives, and how it all connects to the Olympic Site Selection Committee.
  Fact 0.114: Episode 163, 'The Glory That Was ...', is omitted in the DVD release of season 8.
  Fact 0.115: Episode 164, 'Family Values', is season 9 episode 9.
  Fact 0.116: Episode 164, 'Family Values', was directed by Jean de Segonzac.
  Fact 0.117: Episode 164, 'Family Values', has a teleplay by Walon Green.
  Fact 0.118: Episode 164, 'Family Values', has a story by Walon Green and Antoinette Stella.
  Fact 0.119: Episode 164, 'Family Values', first aired in the USA on June 21, 2009.
  Fact 0.120: Episode 164, 'Family Values', aired on NBC on August 3, 2009.
  Fact 0.121: Episode 164, 'Family Values', has production code 08015.
  Fact 0.122: Episode 164, 'Family Values', had 3.44 million U.S. viewers.
  Fact 0.123: When a flier from a local Catholic high school drama production connects three murders, Detectives Goren and Eames are back on the case.
  Fact 0.124: Goren and Eames want to understand the psychology of a religious fanatic before he claims another victim.
  Fact 0.125: Episode 165, 'Salome in Manhattan', is season 10 episode 10.
  Fact 0.126: Episode 165 is also listed under the title 'Salome'.
  Fact 0.127: Episode 165, 'Salome in Manhattan', was directed by Steve Shill.
  Fact 0.128: Episode 165, 'Salome in Manhattan', was written by Andrew Lipsitz.
  Fact 0.129: Episode 165, 'Salome in Manhattan', first aired in the USA on June 28, 2009.
  Fact 0.130: Episode 165, 'Salome in Manhattan', aired on NBC on August 10, 2009.
  Fact 0.131: Episode 165, 'Salome in Manhattan', has production code 08006.
  Fact 0.132: Episode 165, 'Salome in Manhattan', has no U.S. viewer figure (N/A).
  Fact 0.133: A celebrity chef's star is on the rise as he plans the opening of a trendy restaurant with powerful partners.
  Fact 0.134: The chef's tabloid-friendly, gorgeous girlfriend and business partner is killed under mysterious circumstances.
  Fact 0.135: Detectives Nichols and Wheeler enter the seductive world of power and privilege.
  Fact 0.136: Detectives Nichols and Wheeler discover desperation and murder.
  Fact 0.137: The more the detectives learn about the victim and the men in her life, the more they learn about who the true killer is.
  Fact 0.138: Episode 166, 'Lady's Man', is season 11 episode 11.
  Fact 0.139: Episode 166, 'Lady's Man', was directed by Ken Girotti.
  Fact 0.140: Episode 166, 'Lady's Man', was written by Michael S. Chernuchin.
  Fact 0.141: Episode 166, 'Lady's Man', first aired in the USA on June 28, 2009.
  Fact 0.142: Episode 166, 'Lady's Man', aired on NBC on August 17, 2009.
  Fact 0.143: Episode 166, 'Lady's Man', has production code 08011.
  Fact 0.144: Episode 166, 'Lady's Man', has no U.S. viewer figure (N/A).
  Fact 0.145: Detectives Goren and Eames investigate the killing of a man who became famous for dodging a murder conviction because of a mishandled trial.
  Fact 0.146: The man spun his notoriety into a TV show but turned up dead on his birthday.
  Fact 0.147: Eames had worked the case of his first wife's murder 10 years earlier.
  Fact 0.148: The case also involves an ADA (played by Raul Esparza) who has proven a thorn in Eames' side more than once.
  Fact 0.149: Episode 167, 'Passion', is season 12 episode 12.
  Fact 0.150: Episode 167, 'Passion', was directed by Jonathan Herron.
  Fact 0.151: Episode 167, 'Passion', was written by Michael S. Chernuchin.
  Fact 0.152: Episode 167, 'Passion', first aired in the USA on July 12, 2009.
  Fact 0.153: Episode 167, 'Passion', aired on NBC on August 24, 2009.
  Fact 0.154: Episode 167, 'Passion', has production code 08012.
  Fact 0.155: Episode 167, 'Passion', had 3.47 million U.S. viewers.
  Fact 0.156: Egocentric poet Jacob Garrety (played by Will Chase) would do anything to keep his poetry journal, 'The Village Quarterly', afloat.
  Fact 0.157: Foundation president Don McCallum announces he is cutting funding to the journal.
  Fact 0.158: Jacob appeals to his assistant and lover Lauren to use her sexuality to change Don McCallum's mind.
  Fact 0.159: Lauren reluctantly agrees to use her sexuality to change Don McCallum's mind.
  Fact 0.160: While Jacob enjoys the advances of arts patron Sandra Dunbar (played by Sarah Rafferty) at a poetry reading, Lauren carries out Jacob's wish.
  Fact 0.161: Lauren winds up dead on the street.
  Fact 0.162: Detectives Nichols and Wheeler discover the deadly passion that exists behind the arts.
  Fact 0.163: Episode 168, 'All In', is season 13 episode 13.
  Fact 0.164: Episode 168, 'All In', was directed by David Manson.
  Fact 0.165: Episode 168, 'All In', has a teleplay by Pamela Wechsler, Antoinette Stella, and Walon Green.
  Fact 0.166: Episode 168, 'All In', has a story by Pamela Wechsler.
  Fact 0.167: Episode 168, 'All In', first aired in the USA on July 19, 2009.
  Fact 0.168: Episode 168, 'All In', aired on NBC on August 31, 2009.
  Fact 0.169: Episode 168, 'All In', has production code 08013.
  Fact 0.170: Episode 168, 'All In', had 4.14 million U.S. viewers.
  Fact 0.171: When poker genius Josh Snow (played by Aaron Stanford) loses $80,000 in a card game, backer and bookie Lou Cardinale (played by Boris McGiver) forces Josh Snow into collections to work off the losses.
  Fact 0.172: Lou Cardinale gives Josh Snow a .38 and a box of blanks to extract gambling debts.
  Fact 0.173: Lou Cardinale sends along his mistress, Angela (played by Aleksa Palladino), to keep an eye on the money.
  Fact 0.174: When Josh Snow first collects from a guy in the local neighborhood, he shoots a blank at him.
  Fact 0.175: The guy gave Josh Snow twenty thousand dollars.
  Fact 0.176: Josh Snow then collects from Kip McGonagle (played by Robert Leeshock).
  Fact 0.177: Josh Snow fires the gun expecting a blank, but a real bullet kills Kip McGonagle in the street.
  Fact 0.178: Detectives Goren and Eames know Josh Snow from a former case.
  Fact 0.179: The former case is identical in historical plot elements to this episode but has a different actor and character name, from season five's 'Cruise to Nowhere'.
  Fact 0.180: Goren and Eames enter an elaborate game where they must join in the play, or be played.
  Fact 0.181: Episode 169, 'Major Case', is season 14 episode 14.
  Fact 0.182: Episode 169, 'Major Case', was directed by Chris Zalla.
  Fact 0.183: Episode 169, 'Major Case', was written by Andrew Lipsitz.
  Fact 0.184: Episode 169, 'Major Case', first aired in the USA on July 26, 2009.
  Fact 0.185: Episode 169, 'Major Case', aired on NBC on August 31, 2009.
  Fact 0.186: Episode 169, 'Major Case', has production code 08014.
  Fact 0.187: Episode 169, 'Major Case', had 4.63 million U.S. viewers.
  Fact 0.188: A drug dealer is killed the night before she plans to leave town to live on her cousin's farm.
  Fact 0.189: After Detective Megan Wheeler goes into labor, Detective Eames pairs up with Detective Nichols to investigate and find the girl's killer.
  Fact 0.190: Nichols suspects the girl's killer is an old friend (played by Dylan Baker) who works in the medical examiner's office.
  Fact 0.191: Nichols does all he can to prove that the old friend killed the victim, but it could cost Nichols his job in the process.
  Fact 0.192: Episode 169, 'Major Case', is the last episode to feature Julianne Nicholson as Detective Megan Wheeler.
  Fact 0.193: Detective Megan Wheeler is replaced by Kathryn Erbe (Detective Alexandra Eames) for the remainder of season eight.
  Fact 0.194: As of season nine, Detective Megan Wheeler is replaced by Saffron Burrows (Detective Serena Stevens).
  Fact 0.195: Episode 170, 'Alpha Dog', is season 15 episode 15.
  Fact 0.196: Episode 170, 'Alpha Dog', was directed by Norberto Barba.
  Fact 0.197: Episode 170, 'Alpha Dog', was written by Walon Green.
  Fact 0.198: Episode 170, 'Alpha Dog', first aired in the USA on August 2, 2009.
  Fact 0.199: Episode 170, 'Alpha Dog', aired on NBC on September 7, 2009.
  Fact 0.200: Episode 170, 'Alpha Dog', has production code 08007.
  Fact 0.201: Episode 170, 'Alpha Dog', had 4.23 million U.S. viewers.
  Fact 0.202: A hunky posterboy with a million-dollar torso has what others crave: sex appeal.
  Fact 0.203: When the posterboy shows up dead after a night of sex and drugs with an elusive woman, detectives Goren and Eames uncover the sexual paranoia behind the murder.
  Fact 0.204: Episode 171, 'Revolution', is season 16 episode 16.
  Fact 0.205: Episode 171, 'Revolution', was directed by John David Coles.
  Fact 0.206: Episode 171, 'Revolution', was written by Michael S. Chernuchin.
  Fact 0.207: Episode 171, 'Revolution', first aired in the USA on August 9, 2009.
  Fact 0.208: Episode 171, 'Revolution', aired on NBC on September 7, 2009.
  Fact 0.209: Episode 171, 'Revolution', has production code 08016.
  Fact 0.210: Episode 171, 'Revolution', had 4.83 million U.S. viewers.
  Fact 0.211: A revolutionary plans the kidnapping of a powerful banking executive.
  Fact 0.212: The heist goes wrong and the banking executive is killed.
  Fact 0.213: While probing the murder, Nichols and Eames must also prevent a terrorist campaign.
  Fact 0.214: After Episode 171, 'Revolution', Ed Zuckerman and Michael S. Chernuchin leave the show as executive producers.
  Fact 0.215: Ed Zuckerman returned to Law & Order as its executive producer.
  Fact 0.216: Michael S. Chernuchin returned to Law & Order as its co-executive producer.
  Fact 0.217: Walon Green takes over as executive producer and show runner for all episodes.

Retrieved items:
<item 1>
Law & Order: Criminal Intent (season 8)
'' `` Family Values '' `` Major Case '' `` Revolution '' `` Loyalty '' `` To the Boy in the Blue Knit Cap '' See also Episodes Cast Video game Law & Order
franchise Retrieved from `` https://en.wikipedia.org/w/index.php?title=Law_%26_Order:_Criminal_Intent_(season_8)&oldid=815988037 '' Categories : Law & Order :
Criminal Intent episodes 2009 American television seasons Hidden categories : Pages using citations with accessdate and no URL CS1 maint : Unfit url All
articles with dead external links Articles with dead external links from December 2017 Articles with permanently dead external links Pages using deprecated
image syntax Infobox television season articles that use the season name parameter Talk Contents About Wikipedia Wikiquote Français Italiano Nederlands
Srpskohrvatski / српскохрватски Edit links This page was last edited on 18 December 2017 , at 14 : 29 . Text is available under the Creative Commons Attribution
- ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
</item 1>

<item 2>
Law & Order: Criminal Intent (season 8)
There never was a child at the hotel . As they uncover a shocking conspiracy , Goren and Eames learn that the unstable mother ( Piper Perabo ) has been
delusional about the months - ago accidental death of her daughter , prompting her husband ( Luke Kirby ) to exploit her condition in order to continue
receiving financial support from his rich but childless aunt . Partially inspired by the Madeleine McCann and Caylee Anthony cases . 163 8 `` The Glory That Was
... '' Norberto Barba Robert Nathan June 14 , 2009 ( 2009 - 06 - 14 ) ( USA ) July 20 , 2009 ( 2009 - 07 - 20 ) ( NBC ) 08002 4.14 What should have been a
simple meeting turns to murder , leaving Detectives Wheeler and Nichols to question an embassy , a security company , the victim 's relatives , and how it all
connects to the Olympic Site Selection Committee . This episode is omitted in the DVD release of season 8 .
</item 2>

<item 3>
Law & Order: Criminal Intent (season 8)
161 6 `` Astoria Helen '' Norberto Barba Timothy J. Lea May 31 , 2009 ( 2009 - 05 - 31 ) ( USA ) July 6 , 2009 ( 2009 - 07 - 06 ) ( NBC ) 08004 3.97 A charming
con man named Joe Gallagher ( Creighton James ) targets a lonely , single mother named Helen Bramer ( Arija Bareikis ) to try to hit an armored truck carrying
millions of dollars , Bramer worked for the company of the armored truck . Detectives Nichols and Wheeler investigate when one of the con man 's partners dies
weeks later in a bomb blast . Bramer and Gallagher become the prime suspects in the case when a van owner named Frank Stroup ( Domenick Lombardozzi ) is ripped
off and later shot . But the detectives later learn that Bramer and Gallagher are n't the prime suspects when Bramer 's son Kevin gets abducted by Frank Stroup
, and to rescue the boy , Detective Nichols has to put himself in the middle of a hostage situation .
</item 3>

<item 4>
Law & Order: Criminal Intent (season 8)
171 16 `` Revolution '' John David Coles Michael S. Chernuchin August 9 , 2009 ( 2009 - 08 - 09 ) ( USA ) September 7 , 2009 ( 2009 - 09 - 07 ) ( NBC ) 08016
4.83 A revolutionary plans the kidnapping of a powerful banking executive , but the heist goes wrong and the executive is killed . While probing the murder ,
Nichols and Eames must also prevent a terrorist campaign . After this episode , Ed Zuckerman and Michael S. Chernuchin leave the show as executive producers .
Zuckerman returned to Law & Order as its executive producer and Chernuchin as its co-executive producer . Walon Green takes over as executive producer / show
runner for all episodes . Preceded by Season Seven List of Law & Order : Criminal Intent episodes Succeeded by Season Nine References ( edit ) Jump up ^
Littleton , Cynthia ( September 6 , 2000 ) . `` ' Law & Order ' Makes Case For Second Spinoff '' . The Hollywood Reporter . Retrieved April 26 , 2009 .
</item 4>

<item 5>
Law & Order: Criminal Intent (season 8)
TV By the Numbers . Retrieved June 1 , 2011 . Jump up ^ Seidman , Robert ( August 11 , 2009 ) . `` Most Watched week ever for USA '' . TV By the Numbers .
Retrieved June 1 , 2011 . ( hide ) Law & Order : Criminal Intent Characters Police Alexandra Eames Mike Logan Zack Nichols Robert Goren G. Lynn Bishop Carolyn
Barek Megan Wheeler Captain James Deakins Captain Danny Ross Other characters Ron Carver Elizabeth Rodgers Seasons 5 6 7 8 9 10 Notable episodes `` One '' ``
The Good Doctor '' `` Suite Sorrow '' `` Graansha '' `` Happy Family '' `` Stress Position '' `` Weeping Willow '' `` 30 '' `` Amends '' `` Seeds '' ``
Untethered '' `` Senseless '' `` Purgatory '' `` Contract '' `` Assassin '' `` Last Rites '' `` Frame '' `` Rock Star '' `` Faithfully '' `` The Glory That Was
.
</item 5>

<item 6>
Law & Order: Criminal Intent (season 8)
`` She was just feeling her new partner out , which is why she did nothing ? '' `` Playing Dead , '' the season 's first episode featuring Detectives Goren and
Eames , had 4.578 million viewers , over 400,000 more viewers than the following episode , which was Goldblum 's premiere . For the remainder of the season ,
however , episodes featuring Goldblum and Nicholson were watched by more viewers than the episodes featuring D'Onofrio and Erbe , although the viewing figures
for all episodes continued to drop over the following weeks . The May 10 episode was watched by 3.14 million viewers and placed outside the top twenty cable
network television shows for the week , although eight programs that did place inside the top twenty were NBA Playoff matches . Subsequent episodes were watched
by 3.709 million viewers and 4.012 million viewers , but the June 7 episode saw the figures begin to climb . `` The Glory That Was ... , '' which aired on June
14 , was the second - highest - rated cable television program for that night .
</item 6>

<item 7>
Law & Order: Criminal Intent (season 8)
This episode is omitted in the DVD release of season 8 . 164 9 `` Family Values '' Jean de Segonzac Teleplay by : Walon Green Story by : Walon Green &
Antoinette Stella June 21 , 2009 ( 2009 - 06 - 21 ) ( USA ) August 3 , 2009 ( 2009 - 08 - 03 ) ( NBC ) 08015 3.44 When a flier from a local Catholic high school
drama production connects three murders , Detectives Goren and Eames are back on the case to understand the psychology of a religious fanatic before he claims
another victim . 165 10 `` Salome in Manhattan '' `` Salome '' Steve Shill Andrew Lipsitz June 28 , 2009 ( 2009 - 06 - 28 ) ( USA ) August 10 , 2009 ( 2009 - 08
- 10 ) ( NBC ) 08006 N / A A celebrity chef 's star is on the rise as he plans the opening of a trendy restaurant with powerful partners .
</item 7>

<item 8>
Law & Order: Criminal Intent (season 8)
Detectives Nichols and Wheeler discover the deadly passion that exists behind the arts . 168 13 `` All In '' David Manson Teleplay by : Pamela Wechsler ,
Antoinette Stella , & Walon Green Story by : Pamela Wechsler July 19 , 2009 ( 2009 - 07 - 19 ) ( USA ) August 31 , 2009 ( 2009 - 08 - 31 ) ( NBC ) 08013 4.14
When poker genius Josh Snow ( Aaron Stanford ) loses $80,000 in a card game , backer and bookie Lou Cardinale ( Boris McGiver ) forces him into collections to
work off the losses . Lou gives him a . 38 and a box of blanks to extract gambling debts , and sends along his mistress , Angela ( Aleksa Palladino ) , to keep
an eye on the money . When Josh first collects from a guy in the local neighborhood , he shoots a blank at him . The guy gave him twenty thousand dollars and
then he collects from Kip McGonagle ( Robert Leeshock ) .
</item 8>

<item 9>
Law & Order: Criminal Intent (season 8)
It is the second spin - off of the long - running crime drama Law & Order , and was created by Dick Wolf and René Balcer . Law & Order : Criminal Intent follows
the New York City Police Department 's Major Case Squad , which investigates high - profile murder cases . Season eight starred Vincent D'Onofrio as Detective
Robert Goren , Kathryn Erbe as Detective Alexandra Eames , Julianne Nicholson as Detective Megan Wheeler , and Eric Bogosian as Captain Danny Ross . Jeff
Goldblum joined the cast as Detective Zack Nichols . The season was executive produced by the following teams : Walon Green and Michael Chernuchin ; Ed
Zuckerman and Tim Lea ; Dick Wolf and Peter Jankowski ; and Norberto Barba , Diana Son , Julie Martin , and Arthur W. Forney . Contents ( hide ) 1 Production 2
Cast 3 Distribution 4 Reception 5 Episodes 6 References Production ( edit ) Law & Order : Criminal Intent is the third series in the crime drama Law & Order
franchise , which was created by Dick Wolf in 1990 .
</item 9>

<item 10>
Law & Order: Criminal Intent (season 8)
Goren discovers that Stacy and her family moved to Switzerland for a year when she was in high school , and realizes that Stacy 's younger sister Sophie is
actually her daughter by Neil . While questioning Neil and his mother , Camille ( Kathy Baker ) , it is revealed that she had Rick killed because he was
threatening to expose Neil 's family scandal , thwarting his mayoral bid . 157 `` Rock Star '' Bill D'Elia Ed Zuckerman April 26 , 2009 ( 2009 - 04 - 26 ) ( USA
) May 13 , 2009 ( 2009 - 05 - 13 ) ( NBC ) 08008 4.10 Detective Wheeler and her new partner , Detective Zack Nichols , investigate the stabbing of an artist ,
who was found in a neighborhood rife with ethnic tensions .
</item 10>
=== END ITEM id=d9fa583a4aa9dafe ===

=== ITEM id=7e07536d471aae56 ===
Gold sentences and facts:
Sentence 0: Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France `` for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays '' 1906 Joseph John Thomson United Kingdom `` for his theoretical and experimental investigations on the conduction of electricity by gases '' 1907 Albert Abraham Michelson United States Poland `` for his optical precision instruments and the spectroscopic and metrological investigations carried out with their aid '' 1908 Gabriel Lippmann France `` for his method of reproducing colours photographically based on the phenomenon of interference '' 1909 Guglielmo Marconi Italy `` for their contributions to the development of wireless telegraphy '' Karl Ferdinand Braun Germany 1910 Johannes Diderik van der Waals Netherlands `` for his work on the equation of state for gases and liquids '' 1911 Wilhelm Wien Germany `` for his discoveries regarding the laws governing the radiation of heat '' 1912 Nils Gustaf Dalén Sweden `` for his invention of automatic valves designed to be used in combination with gas accumulators in lighthouses and buoys '' 1913 Heike Kamerlingh - Onnes Netherlands `` for his investigations on the properties of matter at low temperatures which led , inter alia , to the production of liquid helium '' 1914 Max von Laue Germany `` For his discovery of the diffraction of X-rays by crystals '' , an important step in the development of X-ray spectroscopy . 1915 William Henry Bragg United Kingdom `` For their services in the analysis of crystal structure by means of X-rays '' , an important step in the development of X-ray crystallography William Lawrence Bragg Australia United Kingdom 1916 Not awarded World War I 1917 Charles Glover Barkla United Kingdom `` For his discovery of the characteristic Röntgen radiation of the elements '' , another important step in the development of X-ray spectroscopy 1918 Max Planck Germany `` for the services he rendered to the advancement of physics by his discovery of energy quanta '' 1919 Johannes Stark Germany `` for his discovery of the Doppler effect in canal rays and the splitting of spectral lines in electric fields '' 1920 Charles Édouard Guillaume Switzerland `` for the service he has rendered to precision measurements in physics by his discovery of anomalies in nickel - steel alloys '' 1921 Albert Einstein Germany Switzerland `` for his services to theoretical physics , and especially for his discovery of the law of the photoelectric effect '' 1922 Niels Bohr Denmark `` for his services in the investigation of the structure of atoms and of the radiation emanating from them '' 1923 Robert Andrews Millikan United States `` for his work on the elementary charge of electricity and on the photoelectric effect '' 1924 Manne Siegbahn Sweden `` for his discoveries and research in the field of X-ray spectroscopy '' 1925 James Franck Germany `` for their discovery of the laws governing the impact of an electron upon an atom '' Gustav Hertz Germany 1926 Jean Baptiste Perrin France `` for his work on the discontinuous structure of matter , and especially for his discovery of sedimentation equilibrium '' 1927 Arthur Holly Compton United States `` for his discovery of the effect named after him '' Charles Thomson Rees Wilson United Kingdom `` for his method of making the paths of electrically charged particles visible by condensation of vapour '' 1928 Owen Willans Richardson United Kingdom `` for his work on the thermionic phenomenon and especially for the discovery of the law named after him '' 1929 Louis Victor Pierre Raymond , 7th Duc de Broglie France `` for his discovery of the wave nature of electrons '' 1930 Chandrasekhara Venkata Raman India `` for his work on the scattering of light and for the discovery of the effect named after him '' 1931 Not awarded 1932 Werner Heisenberg Germany `` for the creation of quantum mechanics , the application of which has , inter alia , led to the discovery of the allotropic forms of hydrogen '' 1933 Erwin Schrödinger Austria `` for the discovery of new productive forms of atomic theory '' Paul Dirac United Kingdom 1934 Not awarded 1935 James Chadwick United Kingdom `` for the discovery of the neutron '' 1936 Victor Francis Hess Austria `` for his discovery of cosmic radiation '' Carl David Anderson United States `` for his discovery of the positron '' 1937 Clinton Joseph Davisson United States `` for their experimental discovery of the diffraction of electrons by crystals '' George Paget Thomson United Kingdom 1938 Enrico Fermi Italy `` for his demonstrations of the existence of new radioactive elements produced by neutron irradiation , and for his related discovery of nuclear reactions brought about by slow neutrons '' 1939 Ernest Lawrence United States `` for the invention and development of the cyclotron and for results obtained with it , especially with regard to artificial radioactive elements '' 1940 Not awarded World War II 1941 Not awarded World War II 1942 Not awarded World War II 1943 Otto Stern United States `` for his contribution to the development of the molecular ray method and his discovery of the magnetic moment of the proton '' 1944 Isidor Isaac Rabi United States Poland `` for his resonance method for recording the magnetic properties of atomic nuclei '' 1945 Wolfgang Pauli Austria `` for the discovery of the Exclusion Principle , also called the Pauli principle '' 1946 Percy Williams Bridgman United States `` for the invention of an apparatus to produce extremely high pressures , and for the discoveries he made there within the field of high pressure physics '' 1947 Edward Victor Appleton United Kingdom `` for his investigations of the physics of the upper atmosphere especially for the discovery of the so - called Appleton layer '' 1948 Patrick Maynard Stuart Blackett United Kingdom `` for his development of the Wilson cloud chamber method , and his discoveries therewith in the fields of nuclear physics and cosmic radiation '' 1949 Hideki Yukawa Japan `` for his prediction of the existence of mesons on the basis of theoretical work on nuclear forces '' 1950 Cecil Frank Powell United Kingdom `` for his development of the photographic method of studying nuclear processes and his discoveries regarding mesons made with this method '' 1951 John Douglas Cockcroft United Kingdom `` for their pioneer work on the transmutation of atomic nuclei by artificially accelerated atomic particles '' Ernest Thomas Sinton Walton Ireland 1952 Felix Bloch Switzerland United States `` for their development of new methods for nuclear magnetic precision measurements and discoveries in connection therewith '' Edward Mills Purcell United States 1953 Frits Zernike Netherlands `` for his demonstration of the phase contrast method , especially for his invention of the phase contrast microscope '' 1954 Max Born West Germany `` for his fundamental research in quantum mechanics , especially for his statistical interpretation of the wavefunction '' Walther Bothe West Germany `` for the coincidence method and his discoveries made therewith '' 1955 Willis Eugene Lamb United States `` for his discoveries concerning the fine structure of the hydrogen spectrum '' Polykarp Kusch United States Germany `` for his precision determination of the magnetic moment of the electron '' 1956 John Bardeen United States `` for their researches on semiconductors and their discovery of the transistor effect '' Walter Houser Brattain United States William Bradford Shockley United States 1957 Tsung - Dao Lee Republic of China `` for their penetrating investigation of the so - called parity laws which has led to important discoveries regarding the elementary particles '' Chen Ning Yang Republic of China 1958 Pavel Alekseyevich Cherenkov Soviet Union `` for the discovery and the interpretation of the Cherenkov effect '' Ilya Frank Soviet Union Igor Yevgenyevich Tamm Soviet Union 1959 Emilio Gino Segrè Italy United States `` for their discovery of the antiproton '' Owen Chamberlain United States 1960 Donald Arthur Glaser United States `` for the invention of the bubble chamber '' 1961 Robert Hofstadter United States `` for his pioneering studies of electron scattering in atomic nuclei and for his thereby achieved discoveries concerning the structure of the nucleons '' Rudolf Ludwig Mössbauer West Germany `` for his researches concerning the resonance absorption of gamma radiation and his discovery in this connection of the effect which bears his name '' 1962 Lev Davidovich Landau Soviet Union `` for his pioneering theories for condensed matter , especially liquid helium '' 1963 Eugene Paul Wigner Hungary United States `` for his contributions to the theory of the atomic nucleus and the elementary particles , particularly through the discovery and application of fundamental symmetry principles '' Maria Goeppert - Mayer United States `` for their discoveries concerning nuclear shell structure '' J. Hans D. Jensen West Germany 1964 Nicolay Gennadiyevich Basov Soviet Union `` for fundamental work in the field of quantum electronics , which has led to the construction of oscillators and amplifiers based on the maser -- laser principle '' Alexander Prokhorov Soviet Union Charles Hard Townes United States 1965 Richard Phillips Feynman United States `` for their fundamental work in quantum electrodynamics ( QED ) , with deep - ploughing consequences for the physics of elementary particles '' Julian Schwinger United States Shin'ichirō Tomonaga Japan 1966 Alfred Kastler France `` for the discovery and development of optical methods for studying Hertzian resonances in atoms '' 1967 Hans Albrecht Bethe United States `` for his contributions to the theory of nuclear reactions , especially his discoveries concerning the energy production in stars '' 1968 Luis Walter Alvarez United States `` for his decisive contributions to elementary particle physics , in particular the discovery of a large number of resonance states , made possible through his development of the technique of using hydrogen bubble chamber and data analysis '' 1969 Murray Gell - Mann United States `` for his contributions and discoveries concerning the classification of elementary particles and their interactions '' 1970 Hannes Olof Gösta Alfvén Sweden `` for fundamental work and discoveries in magneto - hydrodynamics with fruitful applications in different parts of plasma physics '' Louis Néel France `` for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism which have led to important applications in solid state physics '' 1971 Dennis Gabor Hungary United Kingdom `` for his invention and development of the holographic method '' 1972 John Bardeen United States `` for their jointly developed theory of superconductivity , usually called the BCS - theory '' Leon Neil Cooper United States John Robert Schrieffer United States 1973 Leo Esaki Japan `` for their experimental discoveries regarding tunneling phenomena in semiconductors and superconductors , respectively '' Ivar Giaever United States Norway Brian David Josephson United Kingdom `` for his theoretical predictions of the properties of a supercurrent through a tunnel barrier , in particular those phenomena which are generally known as the Josephson effect '' Martin Ryle United Kingdom `` for their pioneering research in radio astrophysics : Ryle for his observations and inventions , in particular of the aperture synthesis technique , and Hewish for his decisive role in the discovery of pulsars '' Antony Hewish United Kingdom Aage Bohr Denmark `` for the discovery of the connection between collective motion and particle motion in atomic nuclei and the development of the theory of the structure of the atomic nucleus based on this connection '' Ben Roy Mottelson Denmark Leo James Rainwater United States 1976 Burton Richter United States `` for their pioneering work in the discovery of a heavy elementary particle of a new kind '' Samuel Chao Chung Ting United States 1977 Philip Warren Anderson United States `` for their fundamental theoretical investigations of the electronic structure of magnetic and disordered systems '' Nevill Francis Mott United Kingdom John Hasbrouck Van Vleck United States 1978 Pyotr Leonidovich Kapitsa Soviet Union `` for his basic inventions and discoveries in the area of low - temperature physics '' Arno Allan Penzias United States `` for their discovery of cosmic microwave background radiation '' Robert Woodrow Wilson United States 1979 Sheldon Lee Glashow United States `` for their contributions to the theory of the unified weak and electromagnetic interaction between elementary particles , including , inter alia , the prediction of the weak neutral current '' Abdus Salam Pakistan Steven Weinberg United States 1980 James Watson Cronin United States `` for the discovery of violations of fundamental symmetry principles in the decay of neutral K - mesons '' Val Logsdon Fitch United States 1981 Nicolaas Bloembergen Netherlands United States `` for their contribution to the development of laser spectroscopy '' Arthur Leonard Schawlow United States Kai Manne Börje Siegbahn Sweden `` for his contribution to the development of high - resolution electron spectroscopy '' 1982 Kenneth G. Wilson United States `` for his theory for critical phenomena in connection with phase transitions '' Subrahmanyan Chandrasekhar India United States `` for his theoretical studies of the physical processes of importance to the structure and evolution of the stars '' William Alfred Fowler United States `` for his theoretical and experimental studies of the nuclear reactions of importance in the formation of the chemical elements in the universe '' 1984 Carlo Rubbia Italy `` for their decisive contributions to the large project , which led to the discovery of the field particles W and Z , communicators of weak interaction '' Simon van der Meer Netherlands 1985 Klaus von Klitzing West Germany `` for the discovery of the quantized Hall effect '' 1986 Ernst Ruska West Germany `` for his fundamental work in electron optics , and for the design of the first electron microscope '' Gerd Binnig West Germany `` for their design of the scanning tunneling microscope '' Heinrich Rohrer Switzerland Johannes Georg Bednorz West Germany `` for their important break - through in the discovery of superconductivity in ceramic materials '' Karl Alexander Müller Switzerland 1988 Leon Max Lederman United States `` for the neutrino beam method and the demonstration of the doublet structure of the leptons through the discovery of the muon neutrino '' Melvin Schwartz United States Jack Steinberger United States 1989 Norman Foster Ramsey United States `` for the invention of the separated oscillatory fields method and its use in the hydrogen maser and other atomic clocks '' Hans Georg Dehmelt United States Germany `` for the development of the ion trap technique '' Wolfgang Paul West Germany 1990 Jerome I. Friedman United States `` for their pioneering investigations concerning deep inelastic scattering of electrons on protons and bound neutrons , which have been of essential importance for the development of the quark model in particle physics '' Henry Way Kendall United States Richard E. Taylor Canada 1991 Pierre - Gilles de Gennes France `` for discovering that methods developed for studying order phenomena in simple systems can be generalized to more complex forms of matter , in particular to liquid crystals and polymers '' 1992 Georges Charpak France Poland `` for his invention and development of particle detectors , in particular the multiwire proportional chamber '' 1993 Russell Alan Hulse United States `` for the discovery of a new type of pulsar , a discovery that has opened up new possibilities for the study of gravitation '' Joseph Hooton Taylor , Jr . United States 1994 Bertram Brockhouse Canada `` for the development of neutron spectroscopy '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' Clifford Glenwood Shull United States `` for the development of the neutron diffraction technique '' and `` for pioneering contributions to the development of neutron scattering techniques for studies of condensed matter '' 1995 Martin Lewis Perl United States `` for the discovery of the tau lepton '' and `` for pioneering experimental contributions to lepton physics '' Frederick Reines United States `` for the detection of the neutrino '' and `` for pioneering experimental contributions to lepton physics '' David Morris Lee United States `` for their discovery of superfluidity in helium - 3 '' Douglas D. Osheroff United States Robert Coleman Richardson United States 1997 Steven Chu United States `` for development of methods to cool and trap atoms with laser light . '' Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands `` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002 Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics , in particular for the detection of cosmic neutrinos '' Masatoshi Koshiba Japan Riccardo Giacconi Italy United States `` for pioneering contributions to astrophysics , which have led to the discovery of cosmic X-ray sources '' 2003 Alexei Alexeyevich Abrikosov Russia United States `` for pioneering contributions to the theory of superconductors and superfluids '' Vitaly Lazarevich Ginzburg Russia Anthony James Leggett United Kingdom United States David J. Gross United States `` for the discovery of asymptotic freedom in the theory of the strong interaction '' Hugh David Politzer United States Frank Wilczek United States 2005 Roy J. Glauber United States `` for his contribution to the quantum theory of optical coherence '' John L. Hall United States `` for their contributions to the development of laser - based precision spectroscopy , including the optical frequency comb technique '' Theodor W. Hänsch Germany 2006 John C. Mather United States `` for their discovery of the blackbody form and anisotropy of the cosmic microwave background radiation '' George F. Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and manipulation of individual quantum systems . '' David J. Wineland United States 2013 François Englert Belgium `` for the theoretical discovery of a mechanism that contributes to our understanding of the origin of mass of subatomic particles , and which recently was confirmed through the discovery of the predicted fundamental particle , by the ATLAS and CMS experiments at CERN 's Large Hadron Collider '' Peter Higgs United Kingdom 2014 Isamu Akasaki Japan `` for the invention of efficient blue light - emitting diodes which has enabled bright and energy - saving white light sources '' Hiroshi Amano Japan Shuji Nakamura Japan United States 2015 Takaaki Kajita Japan `` for the discovery of neutrino oscillations , which shows that neutrinos have mass '' Arthur B. McDonald Canada 2016 David J. Thouless United Kingdom United States `` for theoretical discoveries of topological phase transitions and topological phases of matter '' F. Duncan M. Haldane United Kingdom United States John M. Kosterlitz United Kingdom United States 2017 Rainer Weiss Germany United States `` for decisive contributions to the LIGO detector and the observation of gravitational waves '' Kip Thorne United States Barry Barish United States
  Fact 0.0: 1901: Wilhelm Conrad Röntgen of Germany won the physics prize for the discovery of the remarkable rays later named after him
  Fact 0.1: 1902: Hendrik Lorentz of the Netherlands won the physics prize for research into the influence of magnetism upon radiation phenomena
  Fact 0.2: 1902: Pieter Zeeman of the Netherlands shared the physics prize with Hendrik Lorentz
  Fact 0.3: 1903: Antoine Henri Becquerel of France won the physics prize for the discovery of spontaneous radioactivity
  Fact 0.4: 1903: Pierre Curie of France shared the physics prize for joint research on the radiation phenomena discovered by Henri Becquerel
  Fact 0.5: 1903: Maria Skłodowska-Curie, listed as Poland and France, shared the physics prize for the same joint research on radiation phenomena
  Fact 0.6: 1904: Lord Rayleigh of the United Kingdom won the physics prize for investigating the densities of the most important gases
  Fact 0.7: 1904: Lord Rayleigh of the United Kingdom won the physics prize for discovering argon
  Fact 0.8: 1905: Philipp Eduard Anton von Lenard, listed as Austria-Hungary and Germany, won the physics prize for his work on cathode rays
  Fact 0.9: 1906: Joseph John Thomson of the United Kingdom won the physics prize for theoretical and experimental investigations of electricity conduction by gases
  Fact 0.10: 1907: Albert Abraham Michelson, listed as the United States and Poland, won the physics prize for optical precision instruments
  Fact 0.11: 1907: Albert Abraham Michelson won the physics prize for spectroscopic and metrological investigations carried out with those instruments
  Fact 0.12: 1908: Gabriel Lippmann of France won the physics prize for a method of reproducing colours photographically based on interference
  Fact 0.13: 1909: Guglielmo Marconi of Italy won the physics prize for contributions to the development of wireless telegraphy
  Fact 0.14: 1909: Karl Ferdinand Braun of Germany shared the physics prize for contributions to wireless telegraphy
  Fact 0.15: 1910: Johannes Diderik van der Waals of the Netherlands won the physics prize for work on the equation of state for gases and liquids
  Fact 0.16: 1911: Wilhelm Wien of Germany won the physics prize for discoveries regarding the laws governing the radiation of heat
  Fact 0.17: 1912: Nils Gustaf Dalén of Sweden won the physics prize for inventing automatic valves used with gas accumulators in lighthouses and buoys
  Fact 0.18: 1913: Heike Kamerlingh-Onnes of the Netherlands won the physics prize for investigations of matter at low temperatures, which led to the production of liquid helium
  Fact 0.19: 1914: Max von Laue of Germany won the physics prize for discovering the diffraction of X-rays by crystals
  Fact 0.20: 1915: William Henry Bragg of the United Kingdom shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.21: 1915: William Lawrence Bragg, listed as Australia and the United Kingdom, shared the physics prize for analysing crystal structure by means of X-rays
  Fact 0.22: 1916: No physics prize was awarded because of World War I
  Fact 0.23: 1917: Charles Glover Barkla of the United Kingdom won the physics prize for discovering the characteristic Röntgen radiation of the elements
  Fact 0.24: 1918: Max Planck of Germany won the physics prize for services to physics through his discovery of energy quanta
  Fact 0.25: 1919: Johannes Stark of Germany won the physics prize for discovering the Doppler effect in canal rays and the splitting of spectral lines in electric fields
  Fact 0.26: 1920: Charles Édouard Guillaume of Switzerland won the physics prize for precision measurements in physics through his discovery of anomalies in nickel-steel alloys
  Fact 0.27: 1921: Albert Einstein, listed as Germany and Switzerland, won the physics prize for services to theoretical physics
  Fact 0.28: 1921: Albert Einstein won the physics prize especially for discovering the law of the photoelectric effect
  Fact 0.29: 1922: Niels Bohr of Denmark won the physics prize for investigating the structure of atoms and the radiation emanating from them
  Fact 0.30: 1923: Robert Andrews Millikan of the United States won the physics prize for work on the elementary charge of electricity and on the photoelectric effect
  Fact 0.31: 1924: Manne Siegbahn of Sweden won the physics prize for discoveries and research in X-ray spectroscopy
  Fact 0.32: 1925: James Franck of Germany won the physics prize for discovering the laws governing the impact of an electron upon an atom
  Fact 0.33: 1925: Gustav Hertz of Germany shared the physics prize with James Franck
  Fact 0.34: 1926: Jean Baptiste Perrin of France won the physics prize for work on the discontinuous structure of matter, especially discovering sedimentation equilibrium
  Fact 0.35: 1927: Arthur Holly Compton of the United States won the physics prize for discovering the effect named after him
  Fact 0.36: 1927: Charles Thomson Rees Wilson of the United Kingdom shared the physics prize for a method of making charged particle paths visible by condensation of vapour
  Fact 0.37: 1928: Owen Willans Richardson of the United Kingdom won the physics prize for work on the thermionic phenomenon and discovering the law named after him
  Fact 0.38: 1929: Louis Victor Pierre Raymond, 7th Duc de Broglie, of France won the physics prize for discovering the wave nature of electrons
  Fact 0.39: 1930: Chandrasekhara Venkata Raman of India won the physics prize for work on the scattering of light and discovering the effect named after him
  Fact 0.40: 1931: No physics prize was awarded
  Fact 0.41: 1932: Werner Heisenberg of Germany won the physics prize for creating quantum mechanics, whose application led to the discovery of allotropic forms of hydrogen
  Fact 0.42: 1933: Erwin Schrödinger of Austria won the physics prize for discovering new productive forms of atomic theory
  Fact 0.43: 1933: Paul Dirac of the United Kingdom shared the physics prize for work on atomic theory
  Fact 0.44: 1934: No physics prize was awarded
  Fact 0.45: 1935: James Chadwick of the United Kingdom won the physics prize for discovering the neutron
  Fact 0.46: 1936: Victor Francis Hess of Austria won the physics prize for discovering cosmic radiation
  Fact 0.47: 1936: Carl David Anderson of the United States shared the physics prize for discovering the positron
  Fact 0.48: 1937: Clinton Joseph Davisson of the United States shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.49: 1937: George Paget Thomson of the United Kingdom shared the physics prize for the experimental discovery of electron diffraction by crystals
  Fact 0.50: 1938: Enrico Fermi of Italy won the physics prize for demonstrating new radioactive elements produced by neutron irradiation
  Fact 0.51: 1938: Enrico Fermi won the physics prize for discovering nuclear reactions brought about by slow neutrons
  Fact 0.52: 1939: Ernest Lawrence of the United States won the physics prize for inventing and developing the cyclotron
  Fact 0.53: 1939: Ernest Lawrence won the physics prize for results obtained with the cyclotron, especially on artificial radioactive elements
  Fact 0.54: 1940: No physics prize was awarded because of World War II
  Fact 0.55: 1941: No physics prize was awarded because of World War II
  Fact 0.56: 1942: No physics prize was awarded because of World War II
  Fact 0.57: 1943: Otto Stern of the United States won the physics prize for developing the molecular ray method
  Fact 0.58: 1943: Otto Stern won the physics prize for discovering the magnetic moment of the proton
  Fact 0.59: 1944: Isidor Isaac Rabi, listed as the United States and Poland, won the physics prize for his resonance method for recording the magnetic properties of atomic nuclei
  Fact 0.60: 1945: Wolfgang Pauli of Austria won the physics prize for discovering the Exclusion Principle, also called the Pauli principle
  Fact 0.61: 1946: Percy Williams Bridgman of the United States won the physics prize for inventing an apparatus to produce extremely high pressures
  Fact 0.62: 1946: Percy Williams Bridgman won the physics prize for discoveries he made within high pressure physics
  Fact 0.63: 1947: Edward Victor Appleton of the United Kingdom won the physics prize for investigating the physics of the upper atmosphere
  Fact 0.64: 1947: Edward Victor Appleton won the physics prize especially for discovering the so-called Appleton layer
  Fact 0.65: 1948: Patrick Maynard Stuart Blackett of the United Kingdom won the physics prize for developing the Wilson cloud chamber method
  Fact 0.66: 1948: Patrick Maynard Stuart Blackett won the physics prize for discoveries with that method in nuclear physics and cosmic radiation
  Fact 0.67: 1949: Hideki Yukawa of Japan won the physics prize for predicting the existence of mesons from theoretical work on nuclear forces
  Fact 0.68: 1950: Cecil Frank Powell of the United Kingdom won the physics prize for developing the photographic method of studying nuclear processes
  Fact 0.69: 1950: Cecil Frank Powell won the physics prize for discoveries regarding mesons made with that method
  Fact 0.70: 1951: John Douglas Cockcroft of the United Kingdom shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.71: 1951: Ernest Thomas Sinton Walton of Ireland shared the physics prize for pioneer work on transmuting atomic nuclei with artificially accelerated particles
  Fact 0.72: 1952: Felix Bloch, listed as Switzerland and the United States, shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.73: 1952: Edward Mills Purcell of the United States shared the physics prize for developing new methods for nuclear magnetic precision measurements
  Fact 0.74: 1953: Frits Zernike of the Netherlands won the physics prize for demonstrating the phase contrast method
  Fact 0.75: 1953: Frits Zernike won the physics prize especially for inventing the phase contrast microscope
  Fact 0.76: 1954: Max Born of West Germany won the physics prize for fundamental research in quantum mechanics
  Fact 0.77: 1954: Max Born won the physics prize especially for his statistical interpretation of the wavefunction
  Fact 0.78: 1954: Walther Bothe of West Germany shared the physics prize for the coincidence method and his discoveries made with it
  Fact 0.79: 1955: Willis Eugene Lamb of the United States won the physics prize for discoveries concerning the fine structure of the hydrogen spectrum
  Fact 0.80: 1955: Polykarp Kusch, listed as the United States and Germany, shared the physics prize for precisely determining the magnetic moment of the electron
  Fact 0.81: 1956: John Bardeen of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.82: 1956: Walter Houser Brattain of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.83: 1956: William Bradford Shockley of the United States shared the physics prize for researches on semiconductors and discovering the transistor effect
  Fact 0.84: 1957: Tsung-Dao Lee of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.85: 1957: Chen Ning Yang of the Republic of China shared the physics prize for investigating the so-called parity laws
  Fact 0.86: 1958: Pavel Alekseyevich Cherenkov of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.87: 1958: Ilya Frank of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.88: 1958: Igor Yevgenyevich Tamm of the Soviet Union shared the physics prize for discovering and interpreting the Cherenkov effect
  Fact 0.89: 1959: Emilio Gino Segrè, listed as Italy and the United States, shared the physics prize for discovering the antiproton
  Fact 0.90: 1959: Owen Chamberlain of the United States shared the physics prize for discovering the antiproton
  Fact 0.91: 1960: Donald Arthur Glaser of the United States won the physics prize for inventing the bubble chamber
  Fact 0.92: 1961: Robert Hofstadter of the United States shared the physics prize for pioneering studies of electron scattering in atomic nuclei
  Fact 0.93: 1961: Robert Hofstadter shared the physics prize for discoveries about the structure of the nucleons
  Fact 0.94: 1961: Rudolf Ludwig Mössbauer of West Germany shared the physics prize for researches on the resonance absorption of gamma radiation
  Fact 0.95: 1961: Rudolf Ludwig Mössbauer shared the physics prize for discovering the effect that bears his name
  Fact 0.96: 1962: Lev Davidovich Landau of the Soviet Union won the physics prize for pioneering theories of condensed matter, especially liquid helium
  Fact 0.97: 1963: Eugene Paul Wigner, listed as Hungary and the United States, shared the physics prize for contributions to the theory of the atomic nucleus and elementary particles
  Fact 0.98: 1963: Eugene Paul Wigner shared the physics prize for discovering and applying fundamental symmetry principles
  Fact 0.99: 1963: Maria Goeppert-Mayer of the United States shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.100: 1963: J. Hans D. Jensen of West Germany shared the physics prize for discoveries concerning nuclear shell structure
  Fact 0.101: 1964: Nicolay Gennadiyevich Basov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.102: 1964: Alexander Prokhorov of the Soviet Union shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.103: 1964: Charles Hard Townes of the United States shared the physics prize for fundamental work in quantum electronics on the maser-laser principle
  Fact 0.104: 1965: Richard Phillips Feynman of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.105: 1965: Julian Schwinger of the United States shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.106: 1965: Shin'ichirō Tomonaga of Japan shared the physics prize for fundamental work in quantum electrodynamics
  Fact 0.107: 1966: Alfred Kastler of France won the physics prize for discovering and developing optical methods for studying Hertzian resonances in atoms
  Fact 0.108: 1967: Hans Albrecht Bethe of the United States won the physics prize for contributions to the theory of nuclear reactions
  Fact 0.109: 1967: Hans Albrecht Bethe won the physics prize especially for discoveries about energy production in stars
  Fact 0.110: 1968: Luis Walter Alvarez of the United States won the physics prize for decisive contributions to elementary particle physics
  Fact 0.111: 1968: Luis Walter Alvarez won the physics prize especially for discovering many resonance states using the hydrogen bubble chamber technique
  Fact 0.112: 1969: Murray Gell-Mann of the United States won the physics prize for contributions and discoveries concerning the classification of elementary particles and their interactions
  Fact 0.113: 1970: Hannes Olof Gösta Alfvén of Sweden shared the physics prize for fundamental work in magneto-hydrodynamics
  Fact 0.114: 1970: Louis Néel of France shared the physics prize for fundamental work and discoveries concerning antiferromagnetism and ferrimagnetism
  Fact 0.115: 1971: Dennis Gabor, listed as Hungary and the United Kingdom, won the physics prize for inventing and developing the holographic method
  Fact 0.116: 1972: John Bardeen of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.117: 1972: Leon Neil Cooper of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.118: 1972: John Robert Schrieffer of the United States shared the physics prize for the jointly developed BCS theory of superconductivity
  Fact 0.119: 1973: Leo Esaki of Japan shared the physics prize for experimental discoveries regarding tunneling phenomena in semiconductors
  Fact 0.120: 1973: Ivar Giaever, listed as the United States and Norway, shared the physics prize for experimental discoveries regarding tunneling phenomena in superconductors
  Fact 0.121: 1973: Brian David Josephson of the United Kingdom shared the physics prize for theoretical predictions of supercurrent properties through a tunnel barrier
  Fact 0.122: 1973: Brian David Josephson shared the physics prize for predicting the phenomena known as the Josephson effect
  Fact 0.123: 1974: Martin Ryle of the United Kingdom shared the physics prize for pioneering research in radio astrophysics
  Fact 0.124: 1974: Martin Ryle shared the physics prize for inventing the aperture synthesis technique
  Fact 0.125: 1974: Antony Hewish of the United Kingdom shared the physics prize for his decisive role in discovering pulsars
  Fact 0.126: 1975: Aage Bohr of Denmark shared the physics prize for discovering the connection between collective motion and particle motion in atomic nuclei
  Fact 0.127: 1975: Ben Roy Mottelson of Denmark shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.128: 1975: Leo James Rainwater of the United States shared the physics prize for developing the theory of atomic nucleus structure based on that connection
  Fact 0.129: 1976: Burton Richter of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.130: 1976: Samuel Chao Chung Ting of the United States shared the physics prize for pioneering work in discovering a heavy elementary particle of a new kind
  Fact 0.131: 1977: Philip Warren Anderson of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.132: 1977: Nevill Francis Mott of the United Kingdom shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.133: 1977: John Hasbrouck Van Vleck of the United States shared the physics prize for theoretical investigations of the electronic structure of magnetic and disordered systems
  Fact 0.134: 1978: Pyotr Leonidovich Kapitsa of the Soviet Union shared the physics prize for basic inventions and discoveries in low-temperature physics
  Fact 0.135: 1978: Arno Allan Penzias of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.136: 1978: Robert Woodrow Wilson of the United States shared the physics prize for discovering cosmic microwave background radiation
  Fact 0.137: 1979: Sheldon Lee Glashow of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.138: 1979: Sheldon Lee Glashow shared the physics prize for predicting the weak neutral current
  Fact 0.139: 1979: Abdus Salam of Pakistan shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.140: 1979: Steven Weinberg of the United States shared the physics prize for contributions to the unified weak and electromagnetic interaction
  Fact 0.141: 1980: James Watson Cronin of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.142: 1980: Val Logsdon Fitch of the United States shared the physics prize for discovering violations of fundamental symmetry principles in the decay of neutral K-mesons
  Fact 0.143: 1981: Nicolaas Bloembergen, listed as the Netherlands and the United States, shared the physics prize for contributions to laser spectroscopy
  Fact 0.144: 1981: Arthur Leonard Schawlow of the United States shared the physics prize for contributions to laser spectroscopy
  Fact 0.145: 1981: Kai Manne Börje Siegbahn of Sweden shared the physics prize for contributions to high-resolution electron spectroscopy
  Fact 0.146: 1982: Kenneth G. Wilson of the United States won the physics prize for his theory of critical phenomena in connection with phase transitions
  Fact 0.147: 1982: Subrahmanyan Chandrasekhar, listed as India and the United States, shared the physics prize for theoretical studies of physical processes important to the structure and evolution of stars
  Fact 0.148: 1982: William Alfred Fowler of the United States shared the physics prize for theoretical and experimental studies of nuclear reactions important in forming the chemical elements
  Fact 0.149: 1984: Carlo Rubbia of Italy shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.150: 1984: Simon van der Meer of the Netherlands shared the physics prize for decisive contributions to the large project that discovered the W and Z field particles
  Fact 0.151: 1985: Klaus von Klitzing of West Germany won the physics prize for discovering the quantized Hall effect
  Fact 0.152: 1986: Ernst Ruska of West Germany shared the physics prize for fundamental work in electron optics
  Fact 0.153: 1986: Ernst Ruska shared the physics prize for designing the first electron microscope
  Fact 0.154: 1986: Gerd Binnig of West Germany shared the physics prize for designing the scanning tunneling microscope
  Fact 0.155: 1986: Heinrich Rohrer of Switzerland shared the physics prize for designing the scanning tunneling microscope
  Fact 0.156: 1986: Johannes Georg Bednorz of West Germany shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.157: 1986: Karl Alexander Müller of Switzerland shared the physics prize for a breakthrough in discovering superconductivity in ceramic materials
  Fact 0.158: 1988: Leon Max Lederman of the United States shared the physics prize for the neutrino beam method
  Fact 0.159: 1988: Leon Max Lederman shared the physics prize for demonstrating the doublet structure of the leptons through discovering the muon neutrino
  Fact 0.160: 1988: Melvin Schwartz of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.161: 1988: Jack Steinberger of the United States shared the physics prize for the neutrino beam method and the muon neutrino discovery
  Fact 0.162: 1989: Norman Foster Ramsey of the United States shared the physics prize for inventing the separated oscillatory fields method
  Fact 0.163: 1989: Norman Foster Ramsey shared the physics prize for its use in the hydrogen maser and other atomic clocks
  Fact 0.164: 1989: Hans Georg Dehmelt, listed as the United States and Germany, shared the physics prize for developing the ion trap technique
  Fact 0.165: 1989: Wolfgang Paul of West Germany shared the physics prize for developing the ion trap technique
  Fact 0.166: 1990: Jerome I. Friedman of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.167: 1990: Henry Way Kendall of the United States shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.168: 1990: Richard E. Taylor of Canada shared the physics prize for pioneering investigations of deep inelastic scattering of electrons on protons and bound neutrons
  Fact 0.169: 1991: Pierre-Gilles de Gennes of France won the physics prize for discovering that methods for order phenomena in simple systems generalize to liquid crystals and polymers
  Fact 0.170: 1992: Georges Charpak, listed as France and Poland, won the physics prize for inventing and developing particle detectors
  Fact 0.171: 1992: Georges Charpak won the physics prize especially for the multiwire proportional chamber
  Fact 0.172: 1993: Russell Alan Hulse of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.173: 1993: Joseph Hooton Taylor Jr. of the United States shared the physics prize for discovering a new type of pulsar
  Fact 0.174: 1994: Bertram Brockhouse of Canada shared the physics prize for developing neutron spectroscopy
  Fact 0.175: 1994: Bertram Brockhouse shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.176: 1994: Clifford Glenwood Shull of the United States shared the physics prize for developing the neutron diffraction technique
  Fact 0.177: 1994: Clifford Glenwood Shull shared the physics prize for pioneering contributions to neutron scattering techniques for studies of condensed matter
  Fact 0.178: 1995: Martin Lewis Perl of the United States shared the physics prize for discovering the tau lepton
  Fact 0.179: 1995: Martin Lewis Perl shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.180: 1995: Frederick Reines of the United States shared the physics prize for detecting the neutrino
  Fact 0.181: 1995: Frederick Reines shared the physics prize for pioneering experimental contributions to lepton physics
  Fact 0.182: 1995: David Morris Lee of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.183: 1995: Douglas D. Osheroff of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.184: 1995: Robert Coleman Richardson of the United States shared the physics prize for discovering superfluidity in helium-3
  Fact 0.185: 1997: Steven Chu of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.186: 1997: Claude Cohen-Tannoudji of France shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.187: 1997: William Daniel Phillips of the United States shared the physics prize for developing methods to cool and trap atoms with laser light
  Fact 0.188: 1998: Robert B. Laughlin of the United States shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.189: 1998: Horst Ludwig Störmer of Germany shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.190: 1998: Daniel Chee Tsui, listed as the Republic of China and the United States, shared the physics prize for discovering a new form of quantum fluid with fractionally charged excitations
  Fact 0.191: 1999: Gerard 't Hooft of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.192: 1999: Martinus J.G. Veltman of the Netherlands shared the physics prize for elucidating the quantum structure of electroweak interactions
  Fact 0.193: 2000: Zhores Ivanovich Alferov of Russia shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.194: 2000: Herbert Kroemer of Germany shared the physics prize for developing semiconductor heterostructures used in high-speed and optoelectronics
  Fact 0.195: 2000: Jack St. Clair Kilby of the United States shared the physics prize for his part in the invention of the integrated circuit
  Fact 0.196: 2001: Eric Allin Cornell of the United States shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.197: 2001: Carl Edwin Wieman of the United States shared the physics prize for early fundamental studies of the properties of the condensates
  Fact 0.198: 2001: Wolfgang Ketterle of Germany shared the physics prize for achieving Bose-Einstein condensation in dilute gases of alkali atoms
  Fact 0.199: 2002: Raymond Davis Jr. of the United States shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.200: 2002: Masatoshi Koshiba of Japan shared the physics prize for pioneering contributions to astrophysics through the detection of cosmic neutrinos
  Fact 0.201: 2002: Riccardo Giacconi, listed as Italy and the United States, shared the physics prize for pioneering contributions to astrophysics that led to the discovery of cosmic X-ray sources
  Fact 0.202: 2003: Alexei Alexeyevich Abrikosov, listed as Russia and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.203: 2003: Vitaly Lazarevich Ginzburg of Russia shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.204: 2003: Anthony James Leggett, listed as the United Kingdom and the United States, shared the physics prize for pioneering contributions to the theory of superconductors and superfluids
  Fact 0.205: 2003: David J. Gross of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.206: 2003: Hugh David Politzer of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.207: 2003: Frank Wilczek of the United States shared the physics prize for discovering asymptotic freedom in the theory of the strong interaction
  Fact 0.208: 2005: Roy J. Glauber of the United States won the physics prize for his contribution to the quantum theory of optical coherence
  Fact 0.209: 2005: John L. Hall of the United States shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.210: 2005: Theodor W. Hänsch of Germany shared the physics prize for contributions to laser-based precision spectroscopy, including the optical frequency comb technique
  Fact 0.211: 2006: John C. Mather of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.212: 2006: John C. Mather shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.213: 2006: George F. Smoot of the United States shared the physics prize for discovering the blackbody form of the cosmic microwave background radiation
  Fact 0.214: 2006: George F. Smoot shared the physics prize for discovering the anisotropy of the cosmic microwave background radiation
  Fact 0.215: 2007: Albert Fert of France shared the physics prize for discovering giant magnetoresistance
  Fact 0.216: 2007: Peter Grünberg of Germany shared the physics prize for discovering giant magnetoresistance
  Fact 0.217: 2008: Makoto Kobayashi of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.218: 2008: Toshihide Maskawa of Japan shared the physics prize for discovering the origin of the broken symmetry that predicts at least three families of quarks in nature
  Fact 0.219: 2008: Yoichiro Nambu, listed as Japan and the United States, shared the physics prize for discovering the mechanism of spontaneous broken symmetry in subatomic physics
  Fact 0.220: 2009: Charles K. Kao, listed as Hong Kong, the United Kingdom and the United States, shared the physics prize for groundbreaking achievements concerning light transmission in fibers for optical communication
  Fact 0.221: 2009: Willard S. Boyle, listed as Canada and the United States, shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.222: 2009: George E. Smith of the United States shared the physics prize for inventing an imaging semiconductor circuit, the CCD sensor
  Fact 0.223: 2010: Andre Geim, listed as Russia, the United Kingdom and the Netherlands, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.224: 2010: Konstantin Novoselov, listed as Russia and the United Kingdom, shared the physics prize for groundbreaking experiments regarding the two-dimensional material graphene
  Fact 0.225: 2011: Saul Perlmutter of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.226: 2011: Brian P. Schmidt, listed as Australia and the United States, shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.227: 2011: Adam G. Riess of the United States shared the physics prize for discovering the accelerating expansion of the Universe through observations of distant supernovae
  Fact 0.228: 2012: Serge Haroche of France shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.229: 2012: David J. Wineland of the United States shared the physics prize for ground-breaking experimental methods that enable measuring and manipulating individual quantum systems
  Fact 0.230: 2013: François Englert of Belgium shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.231: 2013: Peter Higgs of the United Kingdom shared the physics prize for the theoretical discovery of a mechanism explaining the origin of mass of subatomic particles
  Fact 0.232: 2014: Isamu Akasaki of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.233: 2014: Hiroshi Amano of Japan shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.234: 2014: Shuji Nakamura, listed as Japan and the United States, shared the physics prize for inventing efficient blue light-emitting diodes
  Fact 0.235: 2015: Takaaki Kajita of Japan shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.236: 2015: Arthur B. McDonald of Canada shared the physics prize for discovering neutrino oscillations, which shows that neutrinos have mass
  Fact 0.237: 2016: David J. Thouless, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.238: 2016: F. Duncan M. Haldane, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.239: 2016: John M. Kosterlitz, listed as the United Kingdom and the United States, shared the physics prize for theoretical discoveries of topological phase transitions and topological phases of matter
  Fact 0.240: 2017: Rainer Weiss, listed as Germany and the United States, shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.241: 2017: Kip Thorne of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
  Fact 0.242: 2017: Barry Barish of the United States shared the physics prize for decisive contributions to the LIGO detector and the observation of gravitational waves
Sentence 1: The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934 , 1940 -- 1942 ) .
  Fact 1.0: The first Nobel Prize in Physics was awarded in 1901
  Fact 1.1: The first Nobel Prize in Physics went to Wilhelm Conrad Röntgen of Germany
  Fact 1.2: Wilhelm Conrad Röntgen received 150,782 SEK
  Fact 1.3: 150,782 SEK is equal to 7,731,004 SEK in December 2007
  Fact 1.4: John Bardeen is the only laureate to win the prize twice
  Fact 1.5: John Bardeen won the prize in 1956 and in 1972
  Fact 1.6: Maria Skłodowska-Curie also won two Nobel Prizes
  Fact 1.7: Maria Skłodowska-Curie won the physics prize in 1903
  Fact 1.8: Maria Skłodowska-Curie won the chemistry prize in 1911
  Fact 1.9: William Lawrence Bragg was, until October 2014, the youngest ever Nobel laureate
  Fact 1.10: William Lawrence Bragg won the prize in 1915 at the age of 25
  Fact 1.11: Two women have won the prize: Curie and Maria Goeppert-Mayer
  Fact 1.12: Maria Goeppert-Mayer won the prize in 1963
  Fact 1.13: As of 2017, the prize has been awarded to 206 individuals
  Fact 1.14: There have been six years in which the Nobel Prize in Physics was not awarded: 1916, 1931, 1934, 1940, 1941 and 1942

Retrieved items:
<item 1>
List of Nobel laureates in Physics
List of Nobel laureates in Physics - wikipedia List of Nobel laureates in Physics Jump to : navigation , search Front side ( obverse ) of the Nobel Prize Medal
for Physics presented to Edward Victor Appleton in 1947 The Nobel Prize in Physics ( Swedish : Nobelpriset i fysik ) is awarded annually by the Royal Swedish
Academy of Sciences to scientists in the various fields of physics . It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel ( who died
in 1896 ) , awarded for outstanding contributions in physics . As dictated by Nobel 's will , the award is administered by the Nobel Foundation and awarded by a
committee that consists of five members elected by the Royal Swedish Academy of Sciences . The award is presented in Stockholm at an annual ceremony on December
10 , the anniversary of Nobel 's death . Each recipient receives a medal , a diploma and a monetary award prize that has varied throughout the years .
</item 1>

<item 2>
List of Nobel laureates in Physics
Yang / T.D. Lee 1958 Cherenkov / Frank / Tamm 1959 Segrè / Chamberlain 1960 Glaser 1961 Hofstadter / Mössbauer 1962 Landau 1963 Wigner / Goeppert - Mayer /
Jensen 1964 Townes / Basov / Prokhorov 1965 Tomonaga / Schwinger / Feynman 1966 Kastler 1967 Bethe 1968 Alvarez 1969 Gell - Mann 1970 Alfvén / Néel 1971 Gabor
1972 Bardeen / Cooper / Schrieffer 1973 Esaki / Giaever / Josephson 1974 Ryle / Hewish 1975 A. Bohr / Mottelson / Rainwater 1976 -- 2000 1976 Richter / Ting
1977 P.W. Anderson / Mott / Van Vleck 1978 Kapitsa / Penzias / R. Wilson 1979 Glashow / Salam / Weinberg 1980 Cronin / Fitch 1981 Bloembergen / Schawlow / K.
</item 2>

<item 3>
List of Nobel laureates in Physics
Siegbahn 1982 K. Wilson 1983 Chandrasekhar / Fowler 1984 Rubbia / Van der Meer 1985 von Klitzing 1986 Ruska / Binnig / Rohrer 1987 Bednorz / Müller 1988
Lederman / Schwartz / Steinberger 1989 Ramsey / Dehmelt / Paul 1990 Friedman / Kendall / R. Taylor 1991 de Gennes 1992 Charpak 1993 Hulse / J. Taylor 1994
Brockhouse / Shull 1995 Perl / Reines 1996 D. Lee / Osheroff / R. Richardson 1997 Chu / Cohen - Tannoudji / Phillips 1998 Laughlin / Störmer / Tsui 1999 ' t
Hooft / Veltman 2000 Alferov / Kroemer / Kilby 2001 -- present 2001 Cornell / Ketterle / Wieman 2002 Davis / Koshiba / Giacconi 2003 Abrikosov / Ginzburg /
Leggett 2004 Gross / Politzer / Wilczek 2005 Glauber / Hall / Hänsch
</item 3>

<item 4>
List of Nobel laureates in Physics
Anderson 1937 Davisson / G.P. Thomson 1938 Fermi 1939 Lawrence 1940 1941 1942 1943 Stern 1944 Rabi 1945 Pauli 1946 Bridgman 1947 Appleton 1948 Blackett 1949
Yukawa 1950 Powell 1951 -- 1975 1951 Cockcroft / Walton 1952 Bloch / Purcell 1953 Zernike 1954 Born / Bothe 1955 Lamb / Kusch 1956 Shockley / Bardeen / Brattain
1957 C.N. Yang / T.D.
</item 4>

<item 5>
List of Nobel laureates in Physics
As of 2017 , the prize has been awarded to 206 individuals . There have been six years in which the Nobel Prize in Physics was not awarded ( 1916 , 1931 , 1934
, 1940 -- 1942 ) . Laureates ( edit ) Year Laureate Country Rationale 1901 Wilhelm Conrad Röntgen Germany `` in recognition of the extraordinary services he has
rendered by the discovery of the remarkable rays subsequently named after him '' 1902 Hendrik Lorentz Netherlands `` in recognition of the extraordinary service
they rendered by their researches into the influence of magnetism upon radiation phenomena '' Pieter Zeeman Netherlands 1903 Antoine Henri Becquerel France ``
for his discovery of spontaneous radioactivity '' Pierre Curie France `` for their joint researches on the radiation phenomena discovered by Professor Henri
Becquerel '' Maria Skłodowska - Curie Poland France 1904 Lord Rayleigh United Kingdom `` for his investigations of the densities of the most important gases and
for his discovery of argon in connection with these studies '' 1905 Philipp Eduard Anton von Lenard Austria - Hungary Germany `` for his work on cathode rays ''
1906 Joseph
</item 5>

<item 6>
List of Nobel laureates in Physics
Claude Cohen - Tannoudji France William Daniel Phillips United States 1998 Robert B. Laughlin United States `` for their discovery of a new form of quantum
fluid with fractionally charged excitations '' Horst Ludwig Störmer Germany Daniel Chee Tsui Republic of China United States 1999 Gerard ' t Hooft Netherlands
`` for elucidating the quantum structure of electroweak interactions in physics '' Martinus J.G. Veltman Netherlands 2000 Zhores Ivanovich Alferov Russia `` for
developing semiconductor heterostructures used in high - speed - and optoelectronics '' Herbert Kroemer Germany Jack St. Clair Kilby United States `` for his
part in the invention of the integrated circuit '' 2001 Eric Allin Cornell United States `` for the achievement of Bose -- Einstein condensation in dilute gases
of alkali atoms , and for early fundamental studies of the properties of the condensates '' Carl Edwin Wieman United States Wolfgang Ketterle Germany 2002
Raymond Davis , Jr . United States `` for pioneering contributions to astrophysics ,
</item 6>

<item 7>
List of Nobel laureates in Physics
Contents ( hide ) 1 Statistics 2 Laureates 3 See also 4 References 4.1 Notes 4.2 Citations 4.3 Sources 5 External links Statistics ( edit ) The first Nobel
Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen , of Germany , who received 150,782 SEK , which is equal to 7,731,004 SEK in December 2007 . John
Bardeen is the only laureate to win the prize twice -- in 1956 and 1972 . Maria Skłodowska - Curie also won two Nobel Prizes , for physics in 1903 and chemistry
in 1911 . William Lawrence Bragg was , until October 2014 , the youngest ever Nobel laureate ; he won the prize in 1915 at the age of 25 . Two women have won
the prize : Curie and Maria Goeppert - Mayer ( 1963 ) . As of 2017 , the prize has been awarded to 206 individuals .
</item 7>

<item 8>
List of Nobel laureates in Physics
Smoot United States 2007 Albert Fert France `` for the discovery of giant magnetoresistance '' Peter Grünberg Germany 2008 Makoto Kobayashi Japan `` for the
discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature '' Toshihide Maskawa Japan Yoichiro
Nambu Japan United States `` for the discovery of the mechanism of spontaneous broken symmetry in subatomic physics '' 2009 Charles K. Kao Hong Kong United
Kingdom United States `` for groundbreaking achievements concerning the transmission of light in fibers for optical communication '' Willard S. Boyle Canada
United States `` for the invention of an imaging semiconductor circuit -- the CCD sensor '' George E. Smith United States Andre Geim Russia United Kingdom
Netherlands `` for groundbreaking experiments regarding the two - dimensional material graphene '' Konstantin Novoselov Russia United Kingdom 2011 Saul
Perlmutter United States `` for the discovery of the accelerating expansion of the Universe through observations of distant supernovae '' Brian P. Schmidt
Australia United States Adam G. Riess United States 2012 Serge Haroche France `` for ground - breaking experimental methods that enable measuring and
manipulation of individual quantum systems . ''
</item 8>

<item 9>
List of Nobel laureates in Physics
Gross / Politzer / Wilczek 2005 Glauber / Hall / Hänsch 2006 Mather / Smoot 2007 Fert / Grünberg 2008 Nambu / Kobayashi / Maskawa 2009 Kao / Boyle / Smith 2010
Geim / Novoselov 2011 Perlmutter / Riess / Schmidt 2012 Wineland / Haroche 2013 Englert / Higgs 2014 Akasaki / Amano / Nakamura 2015 Kajita / McDonald 2016
Thouless / Haldane / Kosterlitz 2017 Weiss / Barish / Thorne Nobel Prizes Lists of Nobel laureates Prizes Chemistry Economics Literature Peace Physics
Physiology or Medicine Laureates by subject Chemistry Economics Literature Peace Physics Physiology or Medicine by criterion African Arab Asian Black Female
Latino and Hispanic Jewish by country Argentine Australian Belgian Bengali Chinese Danish Hungarian Indian Israeli Italian Japanese Korean Liberian Polish
Portuguese Romanian Russian Spanish Swedish Turkish Welsh by religion Christian Jewish Muslim Non-religious by year By year by university affiliation University
affiliation by occupation Head of Government and State Committees and organisations Nobel Foundation Nobel Committees Chemistry Economics Physics Physiology or
Medicine Norwegian Nobel Committee
</item 9>

<item 10>
List of Nobel laureates in Physics
Thomson 1907 Michelson 1908 Lippmann 1909 Marconi / Braun 1910 Van der Waals 1911 Wien 1912 Dalén 1913 Kamerlingh Onnes 1914 Laue 1915 W.L. Bragg / W.H. Bragg
1916 1917 Barkla 1918 Planck 1919 Stark 1920 Guillaume 1921 Einstein 1922 N. Bohr 1923 Millikan 1924 M. Siegbahn 1925 Franck / Hertz 1926 -- 1950 1926 Perrin
1927 Compton / C. Wilson 1928 O. Richardson 1929 De Broglie 1930 Raman 1931 1932 Heisenberg 1933 Schrödinger / Dirac 1934 1935 Chadwick 1936 Hess / C.D.
Anderson 1937 Davisson / G.P.
</item 10>
=== END ITEM id=7e07536d471aae56 ===

=== ITEM id=e5e65aed221b1536 ===
Gold sentences and facts:
Sentence 0: A justice of the peace ( JP ) is a judicial officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying forms of training for JPs .
  Fact 0.0: A justice of the peace (JP) is a judicial officer.
  Fact 0.1: A justice of the peace is a judicial officer of a lower or puisne court.
  Fact 0.2: A justice of the peace is elected or appointed by means of a commission (letters patent).
  Fact 0.3: A justice of the peace is elected or appointed to keep the peace.
  Fact 0.4: In past centuries the term commissioner of the peace was often used with the same meaning as justice of the peace.
  Fact 0.5: Depending on the jurisdiction, justices of the peace dispense summary justice.
  Fact 0.6: Depending on the jurisdiction, justices of the peace merely deal with local administrative applications in common law jurisdictions.
  Fact 0.7: Justices of the peace are appointed or elected from the citizens of the jurisdiction in which they serve.
  Fact 0.8: Justices of the peace are (or were) usually not required to have any formal legal education in order to qualify for the office.
  Fact 0.9: Some jurisdictions have varying forms of training for justices of the peace.

Retrieved items:
<item 1>
Justice of the peace
Justices of the peace are elected every two years to these partisan offices . Massachusetts , Connecticut , Vermont ( edit ) Justices of the peace in the
Commonwealth of Massachusetts are often called on to perform marriages and , especially same - sex marriages , which certain religious officials are not willing
to oversee . Justices of the peace in Connecticut can preside over same - sex marriages . Unlike Massachusetts , Connecticut JPs are not penalized for refusing
to perform such ceremonies . Justices of the peace in Connecticut , Massachusetts , and Vermont have the same general oath - giving powers as a notary public .
Louisiana Are elected to serve six year terms as the judicial authority of a ward or district . Have jurisdiction in civil matters when the amount in dispute
does not exceed $5,000 . Do not have jurisdiction when a title to real estate is involved , when the state or any political subdivision is a defendant , or in
successions or probate matters . Can perform marriage ceremonies Minnesota ( edit ) In Minnesota , the history of justices of the peace dates back hundreds of
years to a time when they were considered to be some of the most powerful officials in terms of the authority that they had to influence change in a community .
</item 1>

<item 2>
Justice of the peace
Additionally , the Justice Court hears cases involving county ordinances ( ordinances enacted by the board of supervisors that apply only to unincorporated
areas ) , civil lawsuits up to a limit of $10,000 , small claims cases up to $2,500 , and issues evictions , called writs of restitution ( after a forcible
detainer or special detainer action ( eviction ) being successfully completed by a landlord ) . Justices of the peace , also called JPs , or Judges of the
Justice Court , are elected in partisan elections for four - year terms from specific districts called precincts . They have the same authority and
responsibility as all other judges in the state with respect to performing marriages , administrating oaths , adhering to the code of judicial conduct , and all
aspects of justice administration . However , Arizona law does not require justices of the peace to be lawyers . Many justices of the peace are not legally
trained , although all are required by the Arizona Supreme Court to complete a course at the Arizona Judicial College . As with JPs , municipal judges in
Arizona are not required to be lawyers . Arkansas ( edit ) This section does not cite any sources .
</item 2>

<item 3>
Justice of the peace
Their judicial powers were replaced by full - time , legally qualified district justices ( now called district judges ) and their quasi-judicial powers by lay
Peace Commissioners . Peace commissioners may sign statutory declarations , and may issue summons and search warrants to the Garda Síochána ( Irish police ) . A
Peace Commissioner can not sign an Affidavit , in Ireland that function is performed by a Commissioner for Oaths . Jamaica ( edit ) A justice of the peace ( JP
) , according to the Ministry of Justice , is a person of unquestionable integrity who seeks to promote and protect the rights of the individual and helps to
provide justice to persons in a particular community . Additionally , the JP serves as a justice in petty court sessions , attends juvenile court sessions ,
issues summonses , considers applications for bail , explains and signs legal documents , sits on licensing panels , and gives counsel / advice . Any Jamaican
citizen that can speak and write English is eligible to become a JP . Any club / organisation / citizen can recommend someone to become JP for a community . JPs
are chosen under the Governor - General 's discretion .
</item 3>

<item 4>
Justice of the peace
Justice of the peace - wikipedia Justice of the peace Jump to : navigation , search This article needs additional citations for verification . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) ( Learn how and when to
remove this template message ) Taos County , New Mexico . Justice of the Peace Montoyo hears a minor case . ( 1941 ) A justice of the peace ( JP ) is a judicial
officer , of a lower or puisne court , elected or appointed by means of a commission ( letters patent ) to keep the peace . In past centuries the term
commissioner of the peace was often used with the same meaning . Depending on the jurisdiction , such justices dispense summary justice or merely deal with
local administrative applications in common law jurisdictions . Justices of the peace are appointed or elected from the citizens of the jurisdiction in which
they serve , and are ( or were ) usually not required to have any formal legal education in order to qualify for the office . Some jurisdictions have varying
forms of training for JPs . Contents ( hide ) 1 History 2 Modern use 2.1 Australia 2.1.
</item 4>

<item 5>
Justice of the peace
A JP has contempt power of $100 and up to 3 days in jail per occurrence . A JP is also authorized to perform marriage ceremonies . Replacement with other courts
( edit ) In many states , the office of justice of the peace has been abolished or transferred to another court , such as the magistrate court . In larger
cities , cases may be heard in a municipal court which has jurisdiction only within that city . Most efforts to abolish the office of justice of the peace have
been led by the American Bar Association , which views non-lawyer judges as no longer necessary , as there are now far more persons with formal legal education
than in the past when justices of the peace were first used . California formerly had justice of the peace courts staffed by lay judges , but began phasing them
out after a landmark 1974 decision in which the Supreme Court of California unanimously held that it was a violation of federal due process ( under the
Fourteenth Amendment to the U.S. Constitution ) to allow a non-lawyer to preside over a criminal trial which could result in incarceration of the defendant .
</item 5>

<item 6>
Justice of the peace
Administrative justice of the peace are court officers appointed by the Minister of Justice , and perform duties such as receiving criminal informations and
issuing warrants . Presiding justice of the peace are appointed by commission under the Great Seal , and can try some criminal matters and issue warrants . They
are appointed from advocates of at least ten years ' standing and serve full - time until the age of 70 . Yukon ( edit ) In Yukon , justices of the peace are
lay officers of the court . They sit in the Justice of the Peace Court , which is part of the Territorial Court of Yukon . Hong Kong ( edit ) In Hong Kong , the
historical functions of justices of the peace have been replaced by full - time , legally qualified magistrates . Nowadays , justices of the peace are
essentially titles of honour given by the Government to community leaders , and to certain officials while they are in their terms of offices . They have no
judicial functions , and their main duties include visiting prisons , institutions for young offenders and drug addicts , psychiatric hospitals , remand homes ,
places of refuge , reception and detention centres , administering statutory declarations , and serving as members of advisory panels .
</item 6>

<item 7>
Justice of the peace
JPs are chosen under the Governor - General 's discretion . Malaysia ( edit ) In Malaysia , justices of the peace have largely been replaced in magistrates '
courts by legally qualified ( first - class ) stipendiary magistrates . However , state governments continue to appoint justices of the peace as honours . In
2004 , some associations of justices of the peace pressed the federal government to allow justices of the peace to sit as second - class magistrates in order to
reduce the backlog of cases in the courts . New Zealand ( edit ) A Justice of the Peace in New Zealand is someone of good stature in the community who is
authorized to witness and sign statutory declarations and affidavits as well as certify documents . They may also perform citizenship ceremonies and act as a
Visiting Justice in prisons the function of which is to record a prisoner 's grievance and decide on its validity . They have certain powers to issue search
warrants , and ( in conjunction with another justice of the peace ) may try minor criminal trials in the district court and exercise powers to remand defendants
in custody , grant bail , and adjourn court hearings .
</item 7>

<item 8>
Justice of the peace
United states ( edit ) In some US states , the justice of the peace is a judge of a court of limited jurisdiction , a magistrate , or a quasi-judicial official
with certain statutory or common law magisterial powers . Some states have special qualifications or unique features for the office . The justice of the peace
typically presides over a court that hears misdemeanor cases , traffic violations , and other petty criminal infractions . The justice of the peace may also
have authority over cases involving small debts , landlord and tenant disputes , or other small claims court proceedings . Proceedings before justices of the
peace are often faster and less formal than the proceedings in other courts . In some jurisdictions a party convicted or found liable before a justice of the
peace may have the right to a trial de novo before the judge of a higher court rather than an appeal strictly considered . A justice of the peace also performs
civil marriages . Arizona ( edit ) A justice of the peace has the same jurisdiction as a municipal magistrate with respect to traffic and misdemeanor cases and
restraining orders , though over cases whose affairs are not contained within the confines of a single municipality .
</item 8>

<item 9>
Justice of the peace
Belgium ( edit ) In Belgium , the justice of the peace ( French : juge de paix , Dutch : vrederechter ) is a law - trained judge who sits the lowest Court of
record with original jurisdiction over small claims ( less than € 2500 ) matters of civil , commercial , and family law , including : tenancy and lease disputes
, neighborhood disputes ( party structures , trespassing , noise pollution ) , apartment buildings , small deceased estates , alimony , guardianships & adoption
, and the commitment of mental patients and oversight of their care . Litigants may opt either for conciliation or trial . Lawyers may act as locum tenens JPs .
Canada ( edit ) In Canada , justices of the peace play a role in the administration of justice at the provincial level . Justices are generally appointed by the
lieutenant governors of Canada 's provinces , and by the commissioners of Canada 's territories , on the advice of their relevant premier or Attorney General .
</item 9>

<item 10>
Justice of the peace
( PLD 2016 Supreme Court 581 ) It is pertinent to note however , as many academics have pointed out , that there is great utility in the appointment of such
justices especially in rural areas where enmity between rival groups can lead to the inability of registration of cognizable offences and biased judicial
proceedings . Singapore ( edit ) A justice of the peace in Singapore derives his powers from statute law . He is appointed by the President of the Republic of
Singapore , under the provisions of section 11 ( l ) of the Subordinate Courts Act ( Cap. 321 ) . The President may revoke the appointment of any justice of the
peace . A newly appointed justice of the peace is required by section 17 of the Subordinate Courts Act , to take the oath of office and allegiance as set out in
the schedule to the Subordinate Courts Act , before exercising the functions of his office . In their role , JPs serve as visiting justices of Singapore prisons
, mediators or referees in the subordinate courts . They may also solemnise marriages in the Registry of Marriages or perform the duties of the magistrate
conferred on them by any written law .
</item 10>
=== END ITEM id=e5e65aed221b1536 ===

=== ITEM id=3b394aa23945eb9b ===
Gold sentences and facts:
Sentence 0: Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
  Fact 0.0: The glenoid cavity is relatively shallow compared to the acetabulum, which is at the hip joint.
  Fact 0.1: The shallow glenoid cavity makes the shoulder joint prone to dislocation (luxation).
  Fact 0.2: Strong glenohumeral ligaments and muscles prevent dislocation in most cases.

Retrieved items:
<item 1>
Glenoid cavity
ISBN 1 - 4051 - 3413 - 5 . This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) References ( edit ) Larson ,
Susan G. ( 2009 ) . `` Evolution of the Hominin Shoulder : Early Homo '' . In Grine , Frederick E. ; Fleagle , John G. ; Leakey , Richard E. The First Humans --
Origin and Early Evolution of the Genus Homo . Springer . doi : 10.1007 / 978 - 1 - 4020 - 9980 - 9 . ISBN 978 - 1 - 4020 - 9979 - 3 . ANATOMY & PHYSIOLOGY :
THE UNITY OF FORM AND FUNCTION , SIXTH EDITION Published by McGraw - Hill Written by Kenneth Saladin
http://www.ucsfhealth.org/conditions/glenoid_labrum_tear/index.html External links ( edit ) Wikimedia Commons has media related to Glenoid cavity . Diagram at
cerrocoso.edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online ,
</item 1>

<item 2>
Glenoid cavity
Strong glenohumeral ligaments and muscles prevents dislocation in most cases . By being so shallow the glenoid cavity allows the shoulder joint to have the
greatest mobility of all joints in the body , allowing 120 degrees of unassisted flexion . Additional range of motion in shoulder flexion ( typically up to 180
degrees in humans ) is also accomplished by the great mobility of the scapula ( shoulder blade ) through a process known as scapulohumeral rhythm . Contents (
hide ) 1 Evolution 1.1 In dinosaurs 2 Additional images 3 See also 4 Notes 5 References 6 References 7 External links Evolution ( edit ) Interpretations of the
fossil remains of Australopithecus africanas ( STS 7 ) and A. afarensis ( AL 288 - 1 ; a.k.a. Lucy ) suggest that the glenoid fossa was oriented more cranially
in these species than in modern humans .
</item 2>

<item 3>
Glenoid cavity
at 16 : 19 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Glenoid cavity
Glenoid cavity - wikipedia Glenoid cavity Jump to : navigation , search Glenoid cavity Costal surface of left scapula . Glenoid cavity shown in red . Glenoid
cavity shown in red . This cavity articulates with the head of the humerus . Details Identifiers Latin cavitas glenoidalis , fossa glenoidalis Dorlands /
Elsevier c_16 / 12220465 TA A02. 4.01. 019 FMA 23275 Anatomical terms of bone ( edit on Wikidata ) The glenoid cavity or glenoid fossa of scapula is a part of
the shoulder . It is a shallow , pyriform articular surface , which is located on the lateral angle of the scapula . It is directed laterally and forward and
articulates with the head of the humerus ; it is broader below than above and its vertical diameter is the longest . This cavity forms the glenohumeral joint
along with the humerus . This type of joint is classified as a synovial , ball and socket joint .
</item 4>

<item 5>
Glenoid cavity
Animation of glenohumeral joint . The muscle shown is supraspinatus muscle . See also ( edit ) Glenopolar angle Notes ( edit ) Jump up ^ The word glenoid is
pronounced / ˈɡliːnɔɪd / or / ˈɡlɛnɔɪd / ( both are common ) and is from Greek : gléne , `` socket '' , reflecting the shoulder joint 's ball - and - socket
form . References ( edit ) Jump up ^ `` Scapulohumeral Rhythm - Physiopedia , universal access to physiotherapy knowledge '' . www.physio-pedia.com . Retrieved
2016 - 09 - 01 . Jump up ^ Larson 2009 , p. 65 Jump up ^ Martin , A.J. ( 2006 ) . Introduction to the Study of Dinosaurs . Second Edition . Oxford , Blackwell
Publishing . pg. 299 - 300 . ISBN 1 - 4051 - 3413 - 5 .
</item 5>

<item 6>
Glenoid cavity
This reflects the importance of overhead limb postures and suggests a retention of arboreal adaptations in these hominoid primates , whereas the lateral
orientation of the glenoid in modern humans reflects the typical lowered position of the arm . In dinosaurs ( edit ) In dinosaurs the main bones of the pectoral
girdle were the scapula ( shoulder blade ) and the coracoid , both of which directly articulated with the clavicle . The place on the scapula where it
articulated with the humerus ( upper bone of the forelimb ) is called the glenoid . The glenoid is important because it defines the range of motion of the
humerus . Additional images ( edit ) Left scapula . Glenoid cavity shown in red . Animation . Glenoid cavity shown in red . Glenoid fossa of right side .
Anterior view of left scapula showing glenoid cavity `` 2 '' Animation of shoulder joint . The muscles shown are subscapularis muscle ( at right ) ,
infraspinatus muscle ( at top left ) , teres minor muscle ( at bottom left ) .
</item 6>

<item 7>
Glenoid cavity
This type of joint is classified as a synovial , ball and socket joint . The humerus is held in place within the glenoid cavity by means of the long head of the
biceps tendon . This tendon originates on the superior margin of the glenoid cavity and loops over the shoulder , bracing humerus against the cavity . The
rotator cuff also reinforces this joint more specifically with the supraspinatus tendon to hold the head of the humerus in the glenoid cavity . The cavity
surface is covered with cartilage in the fresh state ; and its margins , slightly raised , give attachment to a fibrocartilaginous structure , the glenoid
labrum , which deepens the cavity . This cartilage is very susceptible to tearing . When torn , it is most commonly known as a SLAP lesion which is generally
caused by repetitive shoulder movements . Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder
joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
</item 7>

<item 8>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 8>

<item 9>
An ossification center is where cartilage begins transforming into bone: cartilage cells enlarge, matrix calcifies, and cells eventually atrophy. Primary
centers appear first—usually before birth, in long-bone shafts and the bodies of irregular bones; secondary centers generally appear later, often after birth,
in long-bone ends. Some irregular bones have multiple primary centers.

The glenoid cavity is a shallow, pear-shaped socket on the scapula that articulates with the humeral head to
</item 9>

<item 10>
Glenoid cavity
tuberosity ) near wrist ( ulnar notch , styloid process , Lister 's tubercle ) Ulna near elbow ( tuberosity , olecranon , coronoid process , radial notch ,
trochlear notch ) near wrist ( styloid process ) Hand Carpal bones scaphoid lunate triquetral pisiform trapezium trapezoid capitate hamate hamulus Metacarpal
bones 1st 2nd 3rd 4th 5th Phalanges proximal intermediate distal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Glenoid_cavity&oldid=805938319 ''
Categories : Wikipedia articles incorporating text from the 20th edition of Gray 's Anatomy ( 1918 ) Shoulder Scapula Hidden categories : Articles containing
Greek - language text Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk Contents About Wikipedia
Deutsch Español فارسی Français Galego Ido Italiano Magyar Português Svenska ไทย Українська Edit links This page was last edited on 18 October 2017 ,
</item 10>
=== END ITEM id=3b394aa23945eb9b ===

=== ITEM id=e8953fb2dc8d91ff ===
Gold sentences and facts:
Sentence 0: No . Official English title Original Japanese title Original air date English air date 151 `` Sabertooth '' `` Seibātūsu '' ( 剣 咬 の 虎 ( セイバー トゥース ) ) October 6 , 2012 November 17 , 2015 The Legion Platoon is tasked by Zentopia 's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again . Before embarking on their mission , they visit Fairy Tail and make amends with the guild for their previous hostility against them . It is later discovered that Fairy Tail has lost their title as the strongest guild in Fiore to the Saber Tooth guild over the past seven years . Two of the strongest wizards in the guild include a pair of Dragon Slayers named Sting and Rogue , who are obsessed with fighting Natsu and Gajeel , respectively . Meanwhile , Makarov brings Gildarts to their abandoned guildhall and shows him Lumen Histoire , a light that Makarov says is Fairy Tail 's greatest secret . Makarov then tells Gildarts that he has been chosen to become the next master of Fairy Tail . 152 `` And So We Aim for the Top '' `` Soshite Ore - tachi wa Chōjō o Mezasu '' ( そして オレ たち は 顶 上 を 目指す ) October 13 , 2012 November 24 , 2015 Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island . The group meets with Poluchka for medicine to strengthen their magic , but are unsuccessful . Wendy recognizes Poluchka 's voice and scent as those of Grandine , the dragon who raised her . Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her . Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 `` Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction . When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years . They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild , Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power . The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave . 155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray , Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus . As they tour the city , Natsu and Happy encounter the twin dragon slayers , Sting and Rogue , who mock Natsu for not being able to kill Acnologia , and boast of becoming true dragon slayers by killing the dragons who raised them . At midnight , Fairy Tail and the other participating guilds are put through a preliminary event -- a race through a massive floating labyrinth -- designed to narrow the number of participating guilds down to eight teams . 156 `` Sky Labyrinth '' `` Sukai Rabirinsu '' ( 空中 迷宮 ( スカイ ラビリンス ) ) November 10 , 2012 December 22 , 2015 Natsu 's team notices that Wendy has gone missing , so Elfman replaces her to keep them from being eliminated . As they make their way through the enchanted labyrinth , the five encounter various other guilds and steal the maps made by the other guilds to better navigate the maze . They manage to reach the end of the maze and pass the elimination round , confident they have arrived first , only to discover they have barely taken last place . Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace , Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla . To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced , they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars '' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once . Humiliated , Gray swears revenge against Raven Tail and Saber Tooth . Afterwards , Lucy is chosen to fight Raven Tail member Flare in the tournament 's first battle round . 159 `` Lucy vs. Flare '' `` Rūshii vs. Furea '' ( ルーシィ vs. フレア ) December 1 , 2012 January 12 , 2016 Lucy displays tremendous skill in her battle against Flare by being able to summon two spirits simultaneously . As Lucy gains the upper hand , Flare cheats by using her magic hair to hold Alzack and Bisca 's daughter Asuka as a hostage , blackmailing Lucy into not fighting back to keep her from harming Asuka . As Lucy tearfully speaks of Asuka 's danger , Natsu is able to hear it and rushes to Asuka to incinerate Flare 's hair , allowing Lucy to counterattack with her Urano Metoria spell in combination with Gemini . However , before the spell fully activates , Flare 's teammate Obra nullifies Lucy 's magic , causing her to collapse and lose the match . 160 `` Portent '' `` Kyōzui '' ( 凶 瑞 ) December 8 , 2012 January 19 , 2016 Blue Pegasus ' Ren defeats Mermaid Heel 's Araña during the games ' second match , while Saber Tooth 's Orga effortlessly defeats Quatro Cerberus ' War Cry in the third match . For the day 's fourth and final match , Jellal is pitted against Lamia Scale 's Jura , intending to avenge the humiliated Fairy Tail members . He mimics Mystogan 's magic to keep up his disguise , but he finds it ineffective to use against Jura and is forced to use his own magic instead . Before he can cast a spell that will expose his true identity , Ultear uses Merudy 's sensory to force Jellal into submission . Meanwhile , the hospitalized Carla awakens after having a troubling premonition of the future . 161 `` Chariots '' `` Chariotto '' ( 战 车 ( チャリオット ) ) December 15 , 2012 January 26 , 2016 Fairy Tail is undaunted by their poor performance on the first day . As the guild celebrates , they are visited and taunted by Bacchus , Erza 's rival from Quatro Cerberus , who has come to replace War Cry . Carla relays her premonition to Poluchka , saying that she saw Mercurius collapsing around what appeared to be a tearfully singing Lucy . The second day of the tournament begins with a footrace atop a train of moving chariots , causing the participants Natsu , Gajeel , and Sting to suffer from motion sickness as a side effect of their Dragon Slayer magic . As Natsu and Gajeel force themselves forward , Sting inquires about their determination to which Natsu responds that it is as a result of the strength of Fairy Tail 's bonds . Sting eventually throws the race while Natsu and Gajeel manage to finish as sixth and seventh placers , respectively , earning Fairy Tail the audience 's respect . 162 `` Elfman vs. Bacchus '' `` Erufuman vs. Bakkasu '' ( エルフマン vs. バッカス ) December 22 , 2012 September 1 , 2015 Due to a miscomunnication between the king and his knight Arcadios , Elfman is chosen as Bacchus ' opponent instead of Erza . Before the match commences , the two opponents make a wager : if Bacchus wins , he will court Elfman 's sisters , and if Elfman wins , Bacchus ' team name will be changed to `` Quatro Puppy '' for the rest of the games . Bacchus ruthlessly attacks Elfman with his bare palms , who manages to damage Bacchus with his defensive Lizardman transformation . Elfman manages to outlast Bacchus as he exhausts himself , winning the match and forcing Bacchus ' team to be renamed . Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his `` Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts . Jenny then transforms into her battle armor while Mirajane shifts into her Satan Soul `` Sitri '' form and easily defeats Jenny . The final match of the day is announced , with Kagura Mikazuchi of Mermaid Heel pitted against Yukino Aguria of Saber Tooth . Yukino proposes to Kagura that they wager their lives on their battle . Meanwhile , Arcadios is confronted by Darton , the kingdom 's defense minister , about his `` Eclipse Project '' . 164 `` Kagura vs. Yukino '' `` Kagura vs. Yukino '' ( カグラ vs. ユキノ ) January 12 , 2013 September 1 , 2015 Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura . Using her gravity - altering magic , Kagura defeats the two spirits , thus forcing Yukino to use a Gatekey that summons Ophiuchus , which is regarded as the strongest among the Celestial Spirits . However , Kagura slays Ophiuchus with her sword still sheathed and easily defeats Yukino . Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance . Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete '' ( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ; however , Lucy turns down the offer . Meanwhile , Milliana tells Erza that Kagura 's sword , Archenemy , is intended to kill Jellal , and that she and Kagura share the same resentment towards Jellal . Natsu and Happy run after Yukino in an attempt to apologize ; however , Natsu realizes that Yukino had been expelled , and infiltrates the Crocus Garden to fight Jiemma . 166 `` Pandemonium '' `` Pandemoniumu '' ( 伏魔殿 ( パンデモニウム ) ) January 26 , 2013 September 1 , 2015 After Natsu punches out Dobengal , the guild member summoned to fight him , he and Jiemma engage in a duel . However , the guild master 's daughter Minerva appears , ends their duel and , taking Happy hostage , convinces Natsu to settle their dispute in the Grand Magic Games . As Natsu leaves , he states that Saber Tooth will never surpass Fairy Tail because they care for their friends , words that are echoed by Rogue , while Sting is left in awe at Natsu 's strength . The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points . Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent scores depending on the extent of power used . In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei '' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve . For the third match , Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail . Since the opponent is from a suspicious guild , the Fairy Tail members take extra precautions by watching the actions of Ivan Dreyar as well as the other four Raven Tail members . However , as the match commences , the spectators are shocked to see Laxus being beaten by Alexei , although no cautious movements can be seen from the Raven Tail team . It turns out that the two dueling people are just illusions created by Alexei in order for him to have a talk with Laxus . Also , the other Raven Tail members being monitored by Fairy Tail are just lifelike illusions , for in reality they are with Alexei in the battlefield . Alexei reveals himself to be Ivan Dreyar . Unable to interrogate Laxus of the Lumen Histoire 's location , Ivan and the Raven Tail members are forced to fight him . Before their battle begins , Ivan declares that Raven Tail is an anti-Fairy Tail guild . 169 `` Wendy vs. Shelia '' `` Wendi vs. Sheria '' ( ウェンディ vs. シェリア ) February 16 , 2013 September 1 , 2015 After Laxus defeats Ivan and the four Raven Tail elite members , Ivan 's illusions disappear to the shock of everyone in the stadium . The game officials declare Laxus the victor , and Ivan tells him that Lumen Histoire is Fairy Tail 's ' darkness ' . Due to the infraction of the rules , the Raven Tail team is disqualified for the remainder of the Games , and its members are captured by the soldiers . Later , Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off to compete for the final match of the day . As the two exchange attacks , gusts of wind surround the arena since Sherria is revealed to be a Sky God Slayer . Wendy employs an advanced Sky Dragon Slayer attack much to Poluchka 's astonishment , causing Sherria to fall on the ground . Wendy is about to be announced as the winner of the bout when Sherria suddenly stands up with her wounds fully healed . Meanwhile , the Crime Sorcière guild senses the nefarious aura , and Jellal proceeds to the Domus Flau audience , oblivious that he bumped into Doranbolt , who , in turn , wonders what is Mystogan doing in Earthland . 170 `` Small Fists '' `` Chiisana Kobushi '' ( 小さな 拳 ) February 23 , 2013 September 1 , 2015 With Cheria being fully healed and Wendy being almost depleted of magic power , their battle continues , with the former casting a high - level spell towards the latter . However , it seems that Wendy enhances Cheria 's physical strength , causing the attack to gain much force and thus missing Wendy . The two competitors then engage in a hand - to - hand combat , but due to the time limit and neither side is giving up , their battle resulted in a draw , giving both teams with 5 points . Meanwhile , Jellal manages to track the sinister aura among the crowd in the form of a mysterious , hooded person . Jellal is about to confront the aura source , but he is abruptly cornered by Doranbolt , Lahar , and other Rune Knights . Doranbalt then interrogates Jellal , and as his cover was blown , the Rune Knights , as well as Kagura and Milliana glancing nearby , are startled to see the true identity of Mystogan . Yajima convinces the Rune Knights that Mystogan is Jellal 's Edolas counterpart , prompting Lahar to release the fugitive . Meanwhile , Laxus and Makarov have a small conversation regarding Lumen Histoire when Mavis appears and explains that Lumen Histoire is the ' light ' of Fairy Tail . 171 `` Naval Battle '' `` Nabaru Batoru '' ( 海战 ( ナバルバトル ) ) March 2 , 2013 September 1 , 2015 As the jubilant Fairy Tail members celebrate their victories , the other people , such as Sting , Kagura , and Milliana , are contemplating on their own concerns , while Yukino is enlisted as a sergeant under Arcadios ' command . Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others rush to her aid . 172 `` A Parfum for You '' `` Kimi ni Sasageru Parufamu '' ( 君 に 捧げる 香り ( パルファム ) ) March 9 , 2013 September 1 , 2015 While Lucy is hospitalized and taken to the infirmary , the Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so . Later , Makarov announces that the two Fairy Tail teams must merge as a result of the disqualification of Raven Tail and to compensate for the odd number of teams for the battle round . He also reveals that the new team will have the former Team A 's 35 points to start with . The new Fairy Tail team , consisting of Natsu , Gray , Erza , Gajeel , and Laxus , are well - received as they enter the field . The second part of the tournament begins with a tag battle between Blue Pegasus ' Ichiya and `` Rabbit '' and Quatro Puppy 's Rocker and Bacchus . To everyone 's shock , it is revealed that `` Rabbit '' is actually Nichiya in disguise , and after the Exceed is punched by Bacchus , Ichiya bulks up to defeat the Quatro Puppy opponents , awarding Blue Pegasus with 10 points . 173 `` Battle of Dragon Slayers ! '' `` Batoru obu Doragon Sureiyā '' ( バトル ・ オブ ・ ドラゴン スレイヤー ) March 16 , 2013 September 1 , 2015 Mermaid Heel 's Kagura and Milliana and Lamia Scale 's Lyon and Yuka face off for the second tag battle of the games . The two teams struggle to earn a victory , only to have their battle ending in a draw due to the time limit . For the third and final match of the day , the spectators look on in silence as Natsu and Gajeel of Team Fairy Tail and Sting and Rogue of Team Saber Tooth enter the field . As the battle commences , Sting and Rogue were about to take the initiative when they were suddenly attacked by Natsu and Gajeel . The four mages then exchange attacks ; however , the audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers . Unable to keep themselves on par with their adversaries , Sting activates ' White Drive ' while Rogue powers up using ' Shadow Drive ' . 174 `` Four Dragons '' `` Yonin no Doragon '' ( 四 人 の 竜 ( ドラゴン ) ) March 23 , 2013 September 1 , 2015 The tag battle continues , with Sting and Rogue beginning to attack Natsu and Gajeel with their powers being increased . After receiving heavy blows from the Saber Tooth Dragon Slayers , Natsu and Gajeel manages to hold them off , thus forcing Sting and Rogue to activate `` Dragon Force '' on their own will , to the shock of the Fairy Tail audience . Sting decides to finish the Fairy Tail Dragon Slayers alone , for he held a promise to his Exceed friend Lector many years ago . As the three mages fight , the tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel , resulting in the collapse of the arena floor , and after a hand - to - hand combat underground , Sting was seemingly the victor of the match . However , the Fairy Tail duo stands up , comments on Sting 's strength , and argues about his position while attacking , thereby causing Natsu to drag Gajeel on a mining cart and send it away . While Gajeel is left incapacitated due to motion sickness , Natsu challenges the Twin Dragons of Saber Tooth alone , in retribution to what Sting did earlier on them . 175 `` Natsu vs. the Twin Dragons '' `` Natsu vs. Sōryū '' ( ナツ vs. 双竜 ) March 30 , 2013 September 1 , 2015 The tag battle of the Dragon Slayers comes to a close , with Natsu overpowering the combined attacks of Sting and Rogue . The Saber Tooth members watch in dismay , while Jiemma seethes in anger over the defeat of the Twin Dragons , frightening the nearby spectators away . The rest of Fairy Tail celebrates in joy as its team receives 10 points , finally surpassing Team Saber Tooth by a single point . Meanwhile , Gajeel , back to his senses after being dragged by Natsu into a cart earlier , stumbles upon a cave full of dragon skeletons . During that night , Jellal manages to corner the source of the dark aura , standing in disbelief at the identity of the mysterious , hooded person .
  Fact 0.0: Episode 151, 'Sabertooth', has the Japanese title 'Seibatusu' and originally aired on October 6, 2012.
  Fact 0.1: Episode 151 aired in English on November 17, 2015.
  Fact 0.2: The Legion Platoon is tasked by Zentopia's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again.
  Fact 0.3: Before embarking on their mission, the Legion Platoon visits Fairy Tail.
  Fact 0.4: The Legion Platoon makes amends with Fairy Tail for their previous hostility against them.
  Fact 0.5: It is later discovered that Fairy Tail has lost its title as the strongest guild in Fiore to Saber Tooth over the past seven years.
  Fact 0.6: Sting and Rogue are two of the strongest wizards in Fairy Tail.
  Fact 0.7: Sting and Rogue are a pair of Dragon Slayers.
  Fact 0.8: Sting is obsessed with fighting Natsu.
  Fact 0.9: Rogue is obsessed with fighting Gajeel.
  Fact 0.10: Makarov brings Gildarts to their abandoned guildhall.
  Fact 0.11: Makarov shows Gildarts Lumen Histoire, a light that Makarov says is Fairy Tail's greatest secret.
  Fact 0.12: Makarov tells Gildarts that he has been chosen to become the next master of Fairy Tail.
  Fact 0.13: Episode 152, 'And So We Aim for the Top', has the Japanese title 'Soshite Ore-tachi wa Chojo o Mezasu' and originally aired on October 13, 2012.
  Fact 0.14: Episode 152 aired in English on November 24, 2015.
  Fact 0.15: Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island.
  Fact 0.16: The group meets with Poluchka for medicine to strengthen their magic.
  Fact 0.17: The group is unsuccessful in strengthening their magic with the medicine.
  Fact 0.18: Wendy recognizes Poluchka's voice and scent as those of Grandine, the dragon who raised her.
  Fact 0.19: Poluchka reveals herself as the Edolas counterpart of Grandine.
  Fact 0.20: Poluchka gives Wendy written instructions for two advanced spells that Grandine failed to teach her.
  Fact 0.21: Makarov announces Gildarts' succession as master to the rest of the guild.
  Fact 0.22: Makarov finds that Gildarts has gone on a journey, leaving a letter of resignation behind.
  Fact 0.23: In the letter, Gildarts reinstates Laxus into the guild.
  Fact 0.24: In the letter, Gildarts returns mastership to Makarov.
  Fact 0.25: In the letter, Gildarts tells Makarov to make Fairy Tail the strongest guild in Fiore again.
  Fact 0.26: Romeo suggests they participate in the Grand Magic Games.
  Fact 0.27: The Grand Magic Games are an annual festival designed to determine the guild rankings in Fiore.
  Fact 0.28: Episode 153, 'Song of the Stars', has the Japanese title 'Hoshiboshi no Uta' and originally aired on October 20, 2012.
  Fact 0.29: Episode 153 aired in English on December 1, 2015.
  Fact 0.30: The Grand Magic Games are occurring in three months' time.
  Fact 0.31: Natsu and his friends train at a beach resort.
  Fact 0.32: On their second day of training, Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction.
  Fact 0.33: When the wizards arrive, they find a surprise party held by the Celestial Spirit King.
  Fact 0.34: The party celebrates the wizards' return from Sirius Island after seven years.
  Fact 0.35: The wizards spend the day in the spirit world partying with the spirits.
  Fact 0.36: Virgo informs the wizards that one day in the spirit world equals three months in the human world.
  Fact 0.37: The wizards find that the Grand Magic Games are upon them.
  Fact 0.38: Episode 154, 'For All the Time We Missed Each Other', has the Japanese title 'Surechigatta Jikan no Bun dake' and originally aired on October 27, 2012.
  Fact 0.39: Episode 154 aired in English on December 8, 2015.
  Fact 0.40: While mulling over their lost training time, the Fairy Tail wizards receive a message via carrier pigeon.
  Fact 0.41: The message requests a private meeting in the forest nearby.
  Fact 0.42: At the meeting, the wizards meet a runaway Jellal, Ultear, and Merudy.
  Fact 0.43: Jellal, Ultear, and Merudy have formed their own guild, Crime Sorciere.
  Fact 0.44: Crime Sorciere aims to defeat Zeref and any dark guilds that try to use his power.
  Fact 0.45: Jellal, Ultear, and Merudy ask Fairy Tail for help in determining the source of a magical anomaly similar to Zeref's magic.
  Fact 0.46: The magical anomaly has been sensed around the Grand Magic Games arena, Domus Flau.
  Fact 0.47: In exchange, Ultear uses her magic to invoke Fairy Tail's Second Origin in order to double their magic power.
  Fact 0.48: Erza talks with Jellal over his regained memories.
  Fact 0.49: Erza and Jellal nearly kiss.
  Fact 0.50: Jellal decides that he and Erza can not be together.
  Fact 0.51: Jellal lies to Erza that he has a fiancee.
  Fact 0.52: Erza quickly picks up on the lie and respects his decision.
  Fact 0.53: Erza sees Crime Sorciere off as they leave.
  Fact 0.54: Episode 155, 'Crocus, the Flower-Blooming Capital', has the Japanese title 'Hanasaku Miyako: Kurokkasu' and originally aired on November 3, 2012.
  Fact 0.55: Episode 155 aired in English on December 15, 2015.
  Fact 0.56: Natsu, Lucy, Gray, Erza, and Wendy are chosen to represent their guild in the Grand Magic Games.
  Fact 0.57: The Grand Magic Games are to be held in Fiore's capital, Crocus.
  Fact 0.58: As they tour the city, Natsu and Happy encounter the twin dragon slayers, Sting and Rogue.
  Fact 0.59: Sting and Rogue mock Natsu for not being able to kill Acnologia.
  Fact 0.60: Sting and Rogue boast of becoming true dragon slayers by killing the dragons who raised them.
  Fact 0.61: At midnight, Fairy Tail and the other participating guilds are put through a preliminary event.
  Fact 0.62: The preliminary event is a race through a massive floating labyrinth.
  Fact 0.63: The preliminary event is designed to narrow the number of participating guilds down to eight teams.
  Fact 0.64: Episode 156, 'Sky Labyrinth', has the Japanese title 'Sukai Rabirinsu' and originally aired on November 10, 2012.
  Fact 0.65: Episode 156 aired in English on December 22, 2015.
  Fact 0.66: Natsu's team notices that Wendy has gone missing.
  Fact 0.67: Elfman replaces Wendy to keep Natsu's team from being eliminated.
  Fact 0.68: As they make their way through the enchanted labyrinth, the five encounter various other guilds.
  Fact 0.69: The five steal the maps made by the other guilds to better navigate the maze.
  Fact 0.70: The five reach the end of the maze and pass the elimination round.
  Fact 0.71: The five are confident they have arrived first.
  Fact 0.72: The five discover they have barely taken last place.
  Fact 0.73: Happy, Lisanna, and the rest of the guild search for Wendy and Carla.
  Fact 0.74: Happy, Lisanna, and the rest of the guild find Wendy and Carla in the garden of the king's palace, Mercurius.
  Fact 0.75: Wendy and Carla have their magic power depleted.
  Fact 0.76: Episode 157, 'New Guild', has the Japanese title 'Shinki Girudo' and originally aired on November 17, 2012.
  Fact 0.77: Episode 157 aired in English on December 29, 2015.
  Fact 0.78: Wendy's body is weakened as a result of losing her magic power.
  Fact 0.79: Poluchka helps Wendy recover.
  Fact 0.80: Wendy formally requests Elfman to take her place in the Grand Magic Games.
  Fact 0.81: Fairy Tail steps into the arena with a negative audience reception.
  Fact 0.82: Fairy Tail is bolstered by their guild mates and the spirit of Mavis, whom only Fairy Tail guild members can see.
  Fact 0.83: The next four participating teams are Quatro Cerberus, Mermaid Heel, Blue Pegasus, and Lamia Scale.
  Fact 0.84: Raven Tail is a former dark guild led by Makarov's son Ivan.
  Fact 0.85: Raven Tail members confess to attacking Wendy and Carla.
  Fact 0.86: The next team is a second team from Fairy Tail composed of Gajeel, Juvia, Mirajane, Laxus, and Jellal.
  Fact 0.87: Jellal is disguised as Mystogan to help investigate the arena.
  Fact 0.88: The final guild to be announced is Saber Tooth.
  Fact 0.89: The audience cheers joyfully for Saber Tooth's announcement.
  Fact 0.90: All eight teams are announced.
  Fact 0.91: Each team sends out one member to participate in the first contest of the games.
  Fact 0.92: Gray and Juvia represent their teams in the first contest.
  Fact 0.93: Episode 158, 'Night of Shooting Stars', has the Japanese title 'Hoshi Furu Yoru ni' and originally aired on November 24, 2012.
  Fact 0.94: Episode 158 aired in English on January 5, 2016.
  Fact 0.95: The games' first contest, 'Hidden', requires the eight participants to score points by finding and attacking each other.
  Fact 0.96: The contest takes place in a virtual city filled with copies of the participants.
  Fact 0.97: Raven Tail member Nalpudding repeatedly targets Gray.
  Fact 0.98: Gray scores last place in the contest.
  Fact 0.99: Saber Tooth's Rufus scores first by using his Memory-Make magic to attack all the other participants at once.
  Fact 0.100: Gray is humiliated and swears revenge against Raven Tail and Saber Tooth.
  Fact 0.101: Lucy is chosen to fight Raven Tail member Flare in the tournament's first battle round.
  Fact 0.102: Episode 159, 'Lucy vs. Flare', has the Japanese title 'Rushii vs. Furea' and originally aired on December 1, 2012.
  Fact 0.103: Episode 159 aired in English on January 12, 2016.
  Fact 0.104: Lucy displays tremendous skill in her battle against Flare by summoning two spirits simultaneously.
  Fact 0.105: Flare cheats by using her magic hair to hold Alzack and Bisca's daughter, Asuka, as a hostage.
  Fact 0.106: Flare blackmails Lucy into not fighting back to keep Flare from harming Asuka.
  Fact 0.107: Lucy tearfully speaks of Asuka's danger.
  Fact 0.108: Natsu hears Lucy and rushes to Asuka.
  Fact 0.109: Natsu incinerates Flare's hair.
  Fact 0.110: Lucy counterattacks with her Urano Metoria spell in combination with Gemini.
  Fact 0.111: Flare's teammate Obra nullifies Lucy's magic before the spell fully activates.
  Fact 0.112: Lucy collapses and loses the match.
  Fact 0.113: Episode 160, 'Portent', has the Japanese title 'Kyozui' and originally aired on December 8, 2012.
  Fact 0.114: Episode 160 aired in English on January 19, 2016.
  Fact 0.115: Blue Pegasus' Ren defeats Mermaid Heel's Arana during the games' second match.
  Fact 0.116: Saber Tooth's Orga effortlessly defeats Quatro Cerberus' War Cry in the third match.
  Fact 0.117: For the day's fourth and final match, Jellal is pitted against Lamia Scale's Jura.
  Fact 0.118: Jellal intends to avenge the humiliated Fairy Tail members.
  Fact 0.119: Jellal mimics Mystogan's magic to keep up his disguise.
  Fact 0.120: Jellal finds Mystogan's magic ineffective against Jura.
  Fact 0.121: Jellal is forced to use his own magic instead.
  Fact 0.122: Before Jellal can cast a spell that will expose his true identity, Ultear uses Merudy's sensory to force Jellal into submission.
  Fact 0.123: The hospitalized Carla awakens after having a troubling premonition of the future.
  Fact 0.124: Episode 161, 'Chariots', has the Japanese title 'Chariotto' and originally aired on December 15, 2012.
  Fact 0.125: Episode 161 aired in English on January 26, 2016.
  Fact 0.126: Fairy Tail is undaunted by their poor performance on the first day.
  Fact 0.127: As the guild celebrates, they are visited and taunted by Bacchus.
  Fact 0.128: Bacchus is Erza's rival from Quatro Cerberus.
  Fact 0.129: Bacchus has come to replace War Cry.
  Fact 0.130: Carla relays her premonition to Poluchka.
  Fact 0.131: In Carla's premonition, Mercurius is collapsing around what appeared to be a tearfully singing Lucy.
  Fact 0.132: The second day of the tournament begins with a footrace atop a train of moving chariots.
  Fact 0.133: Natsu, Gajeel, and Sting suffer from motion sickness as a side effect of their Dragon Slayer magic.
  Fact 0.134: Natsu responds to Sting that his determination is a result of the strength of Fairy Tail's bonds.
  Fact 0.135: Sting eventually throws the race.
  Fact 0.136: Natsu finishes as sixth place.
  Fact 0.137: Gajeel finishes as seventh place.
  Fact 0.138: Natsu and Gajeel earn Fairy Tail the audience's respect.
  Fact 0.139: Episode 162, 'Elfman vs. Bacchus', has the Japanese title 'Erufuman vs. Bakkasu' and originally aired on December 22, 2012.
  Fact 0.140: Episode 162 aired in English on September 1, 2015.
  Fact 0.141: Due to a miscommunication between the king and his knight Arcadios, Elfman is chosen as Bacchus' opponent instead of Erza.
  Fact 0.142: Before the match, Bacchus and Elfman make a wager.
  Fact 0.143: If Bacchus wins, he will court Elfman's sisters.
  Fact 0.144: If Elfman wins, Bacchus' team name will be changed to 'Quatro Puppy' for the rest of the games.
  Fact 0.145: Bacchus ruthlessly attacks Elfman with his bare palms.
  Fact 0.146: Elfman damages Bacchus with his defensive Lizardman transformation.
  Fact 0.147: Elfman outlasts Bacchus as Bacchus exhausts himself.
  Fact 0.148: Elfman wins the match.
  Fact 0.149: Bacchus' team is renamed.
  Fact 0.150: Natsu finds Wendy, Carla, and Poluchka being kidnapped by a group of masked men.
  Fact 0.151: The masked men claim they were hired to kidnap Lucy by Raven Tail.
  Fact 0.152: The masked men are revealed to have actually been sent by Arcadios.
  Fact 0.153: Arcadios intends to sow discord between the guilds.
  Fact 0.154: Arcadios plots to use Lucy for his 'Eclipse Project'.
  Fact 0.155: Episode 163, 'Mirajane vs. Jenny', has the Japanese title 'Mirajen vs. Jeni' and originally aired on January 5, 2013.
  Fact 0.156: Episode 163 aired in English on September 1, 2015.
  Fact 0.157: The second day of the Grand Magic Games continues.
  Fact 0.158: Mirajane of Fairy Tail Team B fights against Jenny Realight of Team Blue Pegasus.
  Fact 0.159: Mirajane and Jenny are well-known for being magazine models.
  Fact 0.160: Their battle consists almost entirely of a fashion showdown.
  Fact 0.161: Jenny proposes a wager that the loser must pose nude for the Sorcerer Magazine.
  Fact 0.162: Mirajane accepts the wager.
  Fact 0.163: Jenny transforms into her battle armor.
  Fact 0.164: Mirajane shifts into her Satan Soul 'Sitri' form.
  Fact 0.165: Mirajane easily defeats Jenny.
  Fact 0.166: The final match of the day is Kagura Mikazuchi of Mermaid Heel against Yukino Aguria of Saber Tooth.
  Fact 0.167: Yukino proposes to Kagura that they wager their lives on their battle.
  Fact 0.168: Arcadios is confronted by Darton, the kingdom's defense minister, about his 'Eclipse Project'.
  Fact 0.169: Episode 164, 'Kagura vs. Yukino', has the Japanese title 'Kagura vs. Yukino' and originally aired on January 12, 2013.
  Fact 0.170: Episode 164 aired in English on September 1, 2015.
  Fact 0.171: Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura.
  Fact 0.172: Kagura defeats Libra and Pisces using her gravity-altering magic.
  Fact 0.173: Yukino uses a Gatekey that summons Ophiuchus, which is regarded as the strongest among the Celestial Spirits.
  Fact 0.174: Kagura slays Ophiuchus with her sword still sheathed.
  Fact 0.175: Kagura easily defeats Yukino.
  Fact 0.176: Jellal discusses the fact that he has not sensed the evil power similar to Zeref's.
  Fact 0.177: Jellal's discussion takes place after the second day of the Grand Magic Games has ended.
  Fact 0.178: Erza reunites with Milliana.
  Fact 0.179: Milliana is revealed to be a member of Mermaid Heel.
  Fact 0.180: Saber Tooth's master Jiemma is dissatisfied with his guild's performance.
  Fact 0.181: Sting is given another chance.
  Fact 0.182: Jiemma humiliates Yukino for her failure.
  Fact 0.183: Episode 165, 'Hatred at Nightfall', has the Japanese title 'Urami wa Yoru no Tobari ni Tsutsumarete' and originally aired on January 19, 2013.
  Fact 0.184: Episode 165 aired in English on September 1, 2015.
  Fact 0.185: After Yukino is expelled from Saber Tooth, she offers her two golden Gatekeys to Lucy.
  Fact 0.186: Yukino offers the Gatekeys to complete Lucy's collection of the Zodiac Gatekeys.
  Fact 0.187: Yukino states that when the twelve Zodiac Gatekeys are all gathered, a gateway that changes the world will open.
  Fact 0.188: Lucy turns down the offer.
  Fact 0.189: Milliana tells Erza that Kagura's sword, Archenemy, is intended to kill Jellal.
  Fact 0.190: Milliana tells Erza that she and Kagura share the same resentment towards Jellal.
  Fact 0.191: Natsu and Happy run after Yukino in an attempt to apologize.
  Fact 0.192: Natsu realizes that Yukino had been expelled.
  Fact 0.193: Natsu infiltrates the Crocus Garden to fight Jiemma.
  Fact 0.194: Episode 166, 'Pandemonium', has the Japanese title 'Pandemoniumu' and originally aired on January 26, 2013.
  Fact 0.195: Episode 166 aired in English on September 1, 2015.
  Fact 0.196: Natsu punches out Dobengal, the guild member summoned to fight him.
  Fact 0.197: Natsu and Jiemma engage in a duel.
  Fact 0.198: The guild master's daughter, Minerva, appears and ends their duel.
  Fact 0.199: Minerva takes Happy hostage.
  Fact 0.200: Minerva convinces Natsu to settle their dispute in the Grand Magic Games.
  Fact 0.201: As Natsu leaves, he states that Saber Tooth will never surpass Fairy Tail because they care for their friends.
  Fact 0.202: Rogue echoes Natsu's words.
  Fact 0.203: Sting is left in awe at Natsu's strength.
  Fact 0.204: The following day, an event called 'Pandemonium' begins.
  Fact 0.205: In Pandemonium, participants enter a magical citadel filled with 100 monsters of varying strengths.
  Fact 0.206: Participants individually do battle with any number of monsters they choose.
  Fact 0.207: Erza, representing Fairy Tail Team A, requests the 100 monsters as her opponents.
  Fact 0.208: Episode 167, '100 Against 1', has the Japanese title '100 tai 1' and originally aired on February 2, 2013.
  Fact 0.209: Episode 167 aired in English on September 1, 2015.
  Fact 0.210: Erza kills the 100 monsters using various types of armor to match their strengths and weaknesses.
  Fact 0.211: The crowd praises Fairy Tail.
  Fact 0.212: Fairy Tail is awarded 10 points.
  Fact 0.213: The game officials decide to determine the ranks of the remaining participants by means of a device called the 'Magic Power Finder', sometimes abbreviated as MPF.
  Fact 0.214: Participants must attack the Magic Power Finder with their magic power.
  Fact 0.215: The Magic Power Finder calculates and displays participants' equivalent scores based on the extent of power used.
  Fact 0.216: Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis.
  Fact 0.217: Makarov is surprised by Cana's use of the Fairy Glitter spell.
  Fact 0.218: Fairy Tail Team B receives 8 points.
  Fact 0.219: The audience bursts into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams.
  Fact 0.220: The masked leader Alexei tells his subordinates Flare, Obra, Nalpudding, and Kurohebi that it is about time to proceed with their true mission.
  Fact 0.221: Raven Tail somehow got insider information regarding the battle matchups for the day.
  Fact 0.222: Episode 168, 'Laxus vs. Alexei', has the Japanese title 'Rakusasu vs. Arekusei' and originally aired on February 9, 2013.
  Fact 0.223: Episode 168 aired in English on September 1, 2015.
  Fact 0.224: In the battle portion of the Grand Magic Games' third day, Milliana of Mermaid Heel beats Semmes of Quatro Puppy.
  Fact 0.225: Saber Tooth's Rufus swiftly defeats Blue Pegasus' Eve.
  Fact 0.226: Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail.
  Fact 0.227: The Fairy Tail members take extra precautions because Raven Tail is a suspicious guild.
  Fact 0.228: Fairy Tail watches the actions of Ivan Dreyar and the other four Raven Tail members.
  Fact 0.229: The spectators are shocked to see Laxus being beaten by Alexei.
  Fact 0.230: No cautious movements can be seen from the Raven Tail team.
  Fact 0.231: The two dueling people are illusions created by Alexei.
  Fact 0.232: Alexei created the illusions so that he could have a talk with Laxus.
  Fact 0.233: The other Raven Tail members being monitored by Fairy Tail are lifelike illusions.
  Fact 0.234: In reality, the other Raven Tail members are with Alexei on the battlefield.
  Fact 0.235: Alexei reveals himself to be Ivan Dreyar.
  Fact 0.236: Ivan and the Raven Tail members are unable to interrogate Laxus about the Lumen Histoire's location.
  Fact 0.237: Ivan and the Raven Tail members are forced to fight Laxus.
  Fact 0.238: Before their battle begins, Ivan declares that Raven Tail is an anti-Fairy Tail guild.
  Fact 0.239: Episode 169, 'Wendy vs. Shelia', has the Japanese title 'Wendi vs. Sheria' and originally aired on February 16, 2013.
  Fact 0.240: Episode 169 aired in English on September 1, 2015.
  Fact 0.241: Laxus defeats Ivan and the four Raven Tail elite members.
  Fact 0.242: Ivan's illusions disappear to the shock of everyone in the stadium.
  Fact 0.243: The game officials declare Laxus the victor.
  Fact 0.244: Ivan tells Laxus that Lumen Histoire is Fairy Tail's 'darkness'.
  Fact 0.245: Raven Tail is disqualified for the remainder of the Games due to the infraction of the rules.
  Fact 0.246: Raven Tail's members are captured by the soldiers.
  Fact 0.247: Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off for the final match of the day.
  Fact 0.248: Sherria is revealed to be a Sky God Slayer.
  Fact 0.249: Gusts of wind surround the arena as Sherria and Wendy exchange attacks.
  Fact 0.250: Wendy employs an advanced Sky Dragon Slayer attack, much to Poluchka's astonishment.
  Fact 0.251: Sherria falls on the ground.
  Fact 0.252: Wendy is about to be announced as the winner of the bout.
  Fact 0.253: Sherria suddenly stands up with her wounds fully healed.
  Fact 0.254: The Crime Sorciere guild senses the nefarious aura.
  Fact 0.255: Jellal proceeds to the Domus Flau audience.
  Fact 0.256: Jellal is oblivious that he bumped into Doranbolt.
  Fact 0.257: Doranbolt wonders what Mystogan is doing in Earthland.
  Fact 0.258: Episode 170, 'Small Fists', has the Japanese title 'Chiisana Kobushi' and originally aired on February 23, 2013.
  Fact 0.259: Episode 170 aired in English on September 1, 2015.
  Fact 0.260: Cheria is fully healed.
  Fact 0.261: Wendy is almost depleted of magic power.
  Fact 0.262: Cheria casts a high-level spell towards Wendy.
  Fact 0.263: Wendy enhances Cheria's physical strength, causing the attack to miss Wendy.
  Fact 0.264: The two competitors engage in hand-to-hand combat.
  Fact 0.265: Due to the time limit and neither side giving up, their battle results in a draw.
  Fact 0.266: The draw gives both teams 5 points.
  Fact 0.267: Jellal tracks the sinister aura among the crowd in the form of a mysterious, hooded person.
  Fact 0.268: Jellal is about to confront the aura source.
  Fact 0.269: Doranbolt, Lahar, and other Rune Knights corner Jellal.
  Fact 0.270: Doranbolt interrogates Jellal.
  Fact 0.271: Jellal's cover is blown.
  Fact 0.272: The Rune Knights, Kagura and Milliana see the true identity of Mystogan.
  Fact 0.273: Yajima convinces the Rune Knights that Mystogan is Jellal's Edolas counterpart.
  Fact 0.274: Lahar releases the fugitive.
  Fact 0.275: Laxus and Makarov have a small conversation regarding Lumen Histoire.
  Fact 0.276: Mavis appears and explains that Lumen Histoire is the 'light' of Fairy Tail.
  Fact 0.277: Episode 171, 'Naval Battle', has the Japanese title 'Nabaru Batoru' and originally aired on March 2, 2013.
  Fact 0.278: Episode 171 aired in English on September 1, 2015.
  Fact 0.279: The jubilant Fairy Tail members celebrate their victories.
  Fact 0.280: Sting, Kagura, and Milliana are contemplating their own concerns.
  Fact 0.281: Yukino is enlisted as a sergeant under Arcadios' command.
  Fact 0.282: The fourth day of the competitions begins with an event called 'Naval Battle'.
  Fact 0.283: Naval Battle involves participants swimming in a giant water sphere and knocking the other competitors out.
  Fact 0.284: After the other participants fall out of the sphere, a five-minute battle between Lucy and Minerva ensues.
  Fact 0.285: Minerva quickly attacks Lucy and steals the keys of the Celestial Spirit mage.
  Fact 0.286: Lucy refuses to surrender.
  Fact 0.287: After hearing Lucy's intentions, Minerva stops attacking until the five-minute mark is reached.
  Fact 0.288: Minerva resumes attacking Lucy until the game officials decide to halt the match.
  Fact 0.289: The Fairy Tail members snap.
  Fact 0.290: Minerva dangles Lucy out of the sphere.
  Fact 0.291: The other Fairy Tail members rush to Lucy's aid.
  Fact 0.292: Episode 172, 'A Parfum for You', has the Japanese title 'Kimi ni Sasageru Parufamu' and originally aired on March 9, 2013.
  Fact 0.293: Episode 172 aired in English on September 1, 2015.
  Fact 0.294: Lucy is hospitalized and taken to the infirmary.
  Fact 0.295: The Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so.
  Fact 0.296: Makarov announces that the two Fairy Tail teams must merge.
  Fact 0.297: The merger is a result of the disqualification of Raven Tail.
  Fact 0.298: The merger compensates for the odd number of teams for the battle round.
  Fact 0.299: The new team will have the former Team A's 35 points to start with.
  Fact 0.300: The new Fairy Tail team consists of Natsu, Gray, Erza, Gajeel, and Laxus.
  Fact 0.301: The new team is well-received as they enter the field.
  Fact 0.302: The second part of the tournament begins with a tag battle between Blue Pegasus' Ichiya and 'Rabbit' and Quatro Puppy's Rocker and Bacchus.
  Fact 0.303: 'Rabbit' is revealed to be Ichiya in disguise.
  Fact 0.304: Bacchus punches the Exceed.
  Fact 0.305: Ichiya bulks up to defeat the Quatro Puppy opponents.
  Fact 0.306: Blue Pegasus is awarded 10 points.
  Fact 0.307: Episode 173, 'Battle of Dragon Slayers!', has the Japanese title 'Batoru obu Doragon Sureiya' and originally aired on March 16, 2013.
  Fact 0.308: Episode 173 aired in English on September 1, 2015.
  Fact 0.309: Mermaid Heel's Kagura and Milliana face off against Lamia Scale's Lyon and Yuka for the second tag battle of the games.
  Fact 0.310: The second tag battle ends in a draw due to the time limit.
  Fact 0.311: For the third and final match of the day, Natsu and Gajeel of Team Fairy Tail enter the field against Sting and Rogue of Team Saber Tooth.
  Fact 0.312: Sting and Rogue were about to take the initiative when Natsu and Gajeel attacked them.
  Fact 0.313: The audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers.
  Fact 0.314: Sting activates 'White Drive'.
  Fact 0.315: Rogue powers up using 'Shadow Drive'.
  Fact 0.316: Episode 174, 'Four Dragons', has the Japanese title 'Yonin no Doragon' and originally aired on March 23, 2013.
  Fact 0.317: Episode 174 aired in English on September 1, 2015.
  Fact 0.318: Sting and Rogue begin attacking Natsu and Gajeel with their powers increased.
  Fact 0.319: After receiving heavy blows from the Saber Tooth Dragon Slayers, Natsu and Gajeel manage to hold them off.
  Fact 0.320: Sting and Rogue are forced to activate 'Dragon Force' on their own will.
  Fact 0.321: The Fairy Tail audience is shocked by Sting and Rogue activating 'Dragon Force'.
  Fact 0.322: Sting decides to finish the Fairy Tail Dragon Slayers alone.
  Fact 0.323: Sting held a promise to his Exceed friend Lector many years ago.
  Fact 0.324: The tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel.
  Fact 0.325: Sting's attack causes the collapse of the arena floor.
  Fact 0.326: After a hand-to-hand combat underground, Sting is seemingly the victor of the match.
  Fact 0.327: Natsu and Gajeel stand up, comment on Sting's strength, and argue about his position while attacking.
  Fact 0.328: Natsu drags Gajeel on a mining cart and sends it away.
  Fact 0.329: Gajeel is left incapacitated due to motion sickness.
  Fact 0.330: Natsu challenges the Twin Dragons of Saber Tooth alone, in retribution for what Sting did earlier.
  Fact 0.331: Episode 175, 'Natsu vs. the Twin Dragons', has the Japanese title 'Natsu vs. Soryu' and originally aired on March 30, 2013.
  Fact 0.332: Episode 175 aired in English on September 1, 2015.
  Fact 0.333: The tag battle of the Dragon Slayers comes to a close.
  Fact 0.334: Natsu overpowers the combined attacks of Sting and Rogue.
  Fact 0.335: The Saber Tooth members watch in dismay.
  Fact 0.336: Jiemma seethes in anger over the defeat of the Twin Dragons.
  Fact 0.337: Jiemma's anger frightens the nearby spectators away.
  Fact 0.338: The rest of Fairy Tail celebrates in joy as its team receives 10 points.
  Fact 0.339: Fairy Tail surpasses Team Saber Tooth by a single point.
  Fact 0.340: Gajeel, back to his senses after being dragged by Natsu into a cart earlier, stumbles upon a cave full of dragon skeletons.
  Fact 0.341: That night, Jellal manages to corner the source of the dark aura.
  Fact 0.342: Jellal stands in disbelief at the identity of the mysterious, hooded person.

Retrieved items:
<item 1>
Fairy Tail’s second Grand Magic Games team includes Gajeel, Juvia, Mirajane, Laxus, and Jellal in disguise; Saber Tooth completes the lineup. Gray places last
in the “Hidden” contest, while Rufus wins. Jellal’s match against Jura nearly exposes him before Ultear intervenes, and Carla awakens from a troubling
premonition. Cana later wins an event with Mavis’s Fairy Glitter, earning Team
</item 1>

<item 2>
Poluchka reveals she is Edolas Grandine and gives Wendy instructions for two advanced spells. Gildarts leaves, resigns as master, reinstates Laxus, and returns
leadership to Makarov, urging him to make Fairy Tail Fiore’s strongest guild. Romeo proposes entering the Grand Magic Games. During training, Virgo warns that
the Celestial Spirit World is threatened. Crime Sorcière asks Fairy Tail to investigate a Zeref-like magical anomaly at the Games’ arena; Ulte
</item 2>

<item 3>
List of Fairy Tail episodes (season 6)
It contains a single story arc , titled `` Dai Matō Enbu '' ( 大 魔 闘 演武 , lit `` Grand Magic Games '' ) , which adapts material from the beginning of the 31st
through the ending of the 35th volumes of Hiro Mashima 's Fairy Tail manga over 24 episodes . Focusing on Natsu and the others who were frozen in time for seven
years on Tenrou Island , the members return to Magnolia and discover that Saber Tooth is the strongest guild in the Fiore Kingdom . Having become the weakest
guild during the absence of its core members , Fairy Tail decides to enter the Grand Magic Games , an annual competition to decide Fiore 's strongest guild .
The season began airing on October 6 , 2012 on TV Tokyo in Japan and concluded its run on March 30 , 2013 . The first DVD compilation , containing the last two
episodes of the fifth season and the first two episodes of the sixth season , was released by Pony Canyon on March 6 , 2013 .
</item 3>

<item 4>
List of Fairy Tail episodes (season 6)
The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic
Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the
painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza
can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave .
155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray
, Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus .
</item 4>

<item 5>
Season 6 of *Fairy Tail* consists of the 24-episode “Grand Magic Games” arc, adapting manga volumes 31–35. After returning from seven years frozen on Tenrou
Island, Natsu and his companions find Fairy Tail has become Fiore’s weakest guild, while Saber Tooth is strongest. Fairy Tail enters the annual Grand Magic
Games to compete for the title of strongest guild. The season aired in Japan from October 6, 2012, to March 30
</item 5>

<item 6>
List of Fairy Tail episodes (season 6)
The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled
with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests
the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters
using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points .
Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes
abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent
scores depending on the extent of power used .
</item 6>

<item 7>
*Fairy Tail* season 6 adapts the Grand Magic Games arc (2012–13), as Natsu’s guild competes to reclaim its status as Fiore’s strongest. Key events include Erza
defeating 100 monsters, Fairy Tail’s clashes with Raven Tail and Saber Tooth, Lucy’s defeat, and Natsu and Gajeel facing Dragon Slayers Sting and Rogue. The
season’s 24-episode arc received DVD releases, an Animax Asia dub, and new opening
</item 7>

<item 8>
In episode 167, “100 Against 1” (February 2, 2013), the Grand Magic Games’ Pandemonium event challenges participants to fight monsters in a magical citadel.
Erza of Fairy Tail Team A takes on all 100, defeating them with armor suited to their strengths and weaknesses. The crowd cheers, earning Fairy Tail 10 points.
Officials then use the Magic Power Finder (MPF) to rank remaining participants by measuring their magical attacks.
</item 8>

<item 9>
List of Fairy Tail episodes (season 6)
To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as
Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced ,
they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars
'' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to
score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who
scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once .
</item 9>

<item 10>
List of Fairy Tail episodes (season 6)
Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her
. Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of
resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest
guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 ``
Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu
and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of
destruction .
</item 10>
=== END ITEM id=e8953fb2dc8d91ff ===

=== ITEM id=5367a2a349f0a8ef ===
Gold sentences and facts:
Sentence 0: No . Cloverway inc . title Original Japanese and Viz Media titles Episode director ( s ) Writer ( s ) Art director ( s ) Animation director ( s ) Original airdate English airdate ( United States ) 90 `` Star Struck , Bad Luck '' `` Premonition of the Apocalypse : The Mysterious New Guardians Appear '' `` Chikyū hōkai no yokan ? Nazo no shin Senshi shutsugen '' ( 地球 崩壊 の 予感 ? 謎 の 新 戦士 出現 ) Junichi Sato Sukehiro Tomita Kenichi Tajiri Masahiro Andō March 19 , 1994 June 12 , 2000 After experiencing apocalyptic visions , Rei is attacked by a powerful monster called a Daimon that Sailor Moon is unable to fight after it disables her brooch , reversing her transformation . The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the Sailor Team to fend for themselves . Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92 `` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an attractive and supposedly male student of a prestigious local school , with whom they both become infatuated . After Minako is defeated in a racing game , Michiru arrives to pick up Haruka , and the two depart with Usagi and Minako soon following behind them . Once Michiru and Haruka separate at a mechanic 's shop , Usagi and Minako take this as their chance , and Haruka tells them about `` him '' - self until a Daimon arrives to attack Kameda , the shop mechanic . The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down , when Uranus and Neptune arrive to investigate Kameda 's pure heart crystal . When it turns out not to be a talisman they retreat and Sailor Moon defeats the Daimon . Later , the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they 'd told them about ; Michiru arrives with a blonde girl , whom Usagi and Minako initially fail to realize is Haruka ; once they do , Minako falls to the ground in anguish , and Haruka notes that she 'd never said she was male . 93 `` Bad Harmony '' `` Usagi 's Idol : The Graceful Genius Michiru '' `` Usagi no akogare ! Yūbi no tensai Michiru '' ( うさぎ の 憧れ ! 優美 な 天才 みちる ) Hiroki Shibata Megumi Sugihara Kazuyuki Hashimoto Taichi Nakamura April 23 , 1994 June 15 , 2000 When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls , she seeks advice from Michiru Kaioh , who suggests that Usagi and Mamoru go to a violin recital . 94 `` Swept Off Her Feet '' `` Protect the Pure Heart : The Three - Way Battle '' `` Pyua na kokoro wo mamore ! Teki mikata mittsu tomoe ransen '' ( 純 な 心 を 守れ ! 敵 味方 三つ巴 乱戦 ) Noriyo Sasaki Sukehiro Tomita Kenichi Tajiri Ikuko Itō April 30 , 1994 June 16 , 2000 Usagi 's friend , Unazuki , is attacked by a Daimon and has her pure heart crystal stolen . But when Sailor Moon tries to take the crystal back to Unazuki , Uranus and Neptune attempt to steal it in order to claim the Talisman . 95 `` Blinded By Love 's Light '' `` Let Moon Help With Your Love Problems '' `` Koi no otasuke wa Mūn ni omakase '' ( 恋 の おたすけ は ムーン に おまかせ ) Harume Kosaka Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga May 7 , 1994 June 19 , 2000 Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other , but find themselves targeted by a Daimon . 96 `` Lita Borrows Trouble '' `` Coldhearted Uranus : Makoto in Danger '' `` Reikoku na Uranusu ? Makoto no pinchi '' ( 冷酷 な ウラヌス ? まこと の ピンチ ) Kōnosuke Uda Yoji Enokido Kazuhisa Asai Masahiro Andō May 14 , 1994 June 20 , 2000 Makoto is targeted by a Daimon , but manages to temporarily evade it . Keen to analyze her pure heart crystal , Haruka befriends Makoto in order to get close to her . 97 `` Damp Spirits '' `` The Labyrinth of Water : Ami Targeted '' `` Mizu no rabarinsu ! Nerawareta Ami '' ( 水 の ラビリンス ! ねら われ た 亜美 ) Junichi Sato Megumi Sugihara Kazuyuki Hashimoto Katsumi Tamegai May 21 , 1994 June 21 , 2000 Ami 's desire to avoid competition leads her to offending Michiru inadvertently in a swimming race , as well as Usagi accidentally hurting Ami 's feelings . Meanwhile , a Daimon selects Ami as a target . 98 `` Friendly Foes '' `` To Save Our Friends : Moon and Uranus Join Forces '' `` Tomodachi wo sukue ! Mūn Uranusu rengō '' ( 友達 を 救え ! ムーン ウラヌス 連合 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Taichi Nakamura May 28 , 1994 June 22 , 2000 After Sailor Neptune is captured and possibly killed by Kaolinite , Sailor Moon and Sailor Uranus find themselves shackled together and forced to reassess their less - than - friendly attitudes toward each other in order to save Sailor Neptune from further harm . 99 `` Mixed Emotions '' `` A Man 's Kindness : Yuichiro Heartbroken by Rei '' `` Otoko no yasashisa ! Yūichirō Rei ni shitsuren ? '' ( 男 の 優し さ ! 雄一郎 レイ に 失恋 ? ) Harume Kosaka Megumi Sugihara Yoshiyuki Shikano Shinya Hasegawa June 18 , 1994 June 23 , 2000 Worried by Rei 's intensive meditation , Yuuichirou becomes concerned that she has fallen deeply head - over-heels in love for another man . When he sees her with Haruka and mistakes her for a man , Yuuichirou is convinced that `` he '' is Rei 's new romantic suitor . 100 `` Individual Happiness '' `` I Want to Quit Being a Sailor Guardian : Minako 's Dilemma '' `` Sērā Senshi wo yametai ! ? Minako no nayami '' ( S 戦士 を 辞め たい ! ? 美奈子 の 悩み ) Hiroki Shibata Yoji Enokido Kazuhisa Asai Mari Tominaga June 25 , 1994 June 26 , 2000 Minako is despondent about her double life of being a Sailor Guardian and a normal girl . Meanwhile , an old friend of hers , who she once fell in love with , is attacked by a Daimon , and Minako , as Sailor Venus , must use her skills of volleyball to save him . 101 `` Birthday Blues , Part 1 '' `` Usagi in Tears : a Glass Slipper for My Birthday '' `` Usagi namida ! Tanjōbi ni garasu no kutsu wo '' ( うさぎ 涙 ! 誕生 日 に ガラス の 靴 を ) Yūji Endō Katsuyuki Sumisawa Kazuyuki Hashimoto Masahiro Andō July 2 , 1994 June 27 , 2000 Usagi is furious when Mamoru forgets her 15th birthday . When Mamoru buys her an expensive pair of glass slippers to make up for it , he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal . To make matters worse , Kaolinite steals Usagi 's Cosmic Heart Compact leaving her unable to transform , and Tuxedo Mask gets captured . 102 `` Birthday Blues , Part Two '' `` The Stolen Pure Heart : Usagi 's Crisis '' `` Ubawareta pyua na kokoro ! Usagi zettai zetsumei '' ( 奪 われ た 純 な 心 ! うさぎ 絶 体 絶命 ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Taichi Nakamura July 16 , 1994 June 28 , 2000 With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped , Usagi and the four other Sailor Guardians , along with Sailors Uranus and Neptune , go to retake the compact so Usagi can become Sailor Moon again , rescue Tuxedo Mask and face Kaolinite in a final showdown at Tokyo Tower . 103 `` Hello , Sailor Mini Moon '' `` The Arrival of a Tiny Pretty Guardian '' `` Yatte kita chiccha na Bishōjo Senshi '' ( やって来 た ちっちゃな 美 少女 戦士 ) Kunihiko Ikuhara Shigeru Yanagawa Yoshiyuki Shikano Kazuya Kuroda August 6 , 1994 June 29 , 2000 Eudial , and the Witches 5 are tasked to take over from Kaolinite 's position , in which they target a genius drummer recruited to play at a festival for Rei . The Sailor Guardians can not defeat the Daimon who attacks her , but Chibiusa arrives as the Guardians ' newest ally , Sailor Chibi - Moon . 104 `` Tainted Tea Party '' `` Making New Friends : Chibi Moon 's Adventure '' `` Tomodachi wo motomete ! Chibi Mūn no katsuyaku '' ( 友達 を 求め て ! ちび ムーン の 活躍 ) Junichi Sato Megumi Sugihara Kazuhisa Asai Ikuko Itō August 20 , 1994 June 30 , 2000 Chibiusa , who has returned from Crystal Tokyo for further training , attempts to adapt to her new home by making friends her age . She tries to befriend a master who does tea ceremonies , only to become the Death Busters ' latest target . 105 `` People Who Need People '' `` I Want Power : Mako Lost in Doubt '' `` Pawā ga hoshī ! Mako - chan no mayoi michi '' ( 力 が 欲しい ! まこ ちゃん の 迷い 道 ) Hiroki Shibata Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga August 27 , 1994 July 3 , 2000 Makoto is training in the mountains , trying to become much stronger after being easily defeated by a Daimon before . Her trainer is attacked , and tells her that nobody can live alone . She attacks the Daimon , fails , and realizes she needs her best friends in order to win . 106 `` Related By Destiny '' `` The Bond of Destiny : Uranus 's Distant Past '' `` Unmei no kizuna ! Uranusu no tōi hi '' ( 運命 の きずな ! ウラヌス の 遠い 日 ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai September 3 , 1994 July 4 , 2000 Haruka remembers the past when she first met Michiru , who told her that she was destined to become Sailor Uranus . Haruka initially refused that destiny , for she loved her life too much and was afraid of change . Haruka was attacked , and Sailor Neptune was injured while defending her . In her anguish , Haruka took the transformation pen and accepted her destiny as Sailor Uranus , the Sailor Guardian of the Sky . 107 `` Art Appreciation '' `` Art Is an Explosion of Love : Chibiusa 's First Love '' `` Geijutsu wa ai no bakuhatsu ! Chibiusa no hatsukoi '' ( 芸術 は 愛 の 爆発 ! ちびうさ の 初恋 ) Yūji Endō Katsuyuki Sumisawa Minoru Ōkōchi Taichi Nakamura September 10 , 1994 July 5 , 2000 Chibiusa meets a boy named Masanori at her art class and develops a crush on him . Unfortunately , he already has a crush on Michiru , who is taking a class at that school as well . When a Daimon attacks Masanori , Chibiusa must take a stand to protect him . 108 `` Everything 's Coming Up Rosey '' `` Usagi Dancing to the Waltz '' `` Usagi no dansu wa warutsu ni notte '' ( うさぎ の ダンス は ワルツ に 乗っ て ) Harume Kosaka Megumi Sugihara Kazuhisa Asai Masahiro Andō September 17 , 1994 July 6 , 2000 An English gentleman named Edwards invites Mamoru and the girls to a fancy party , where Usagi accidentally becomes drunk and the team is forced to confront a fashionable Eudial and her latest Daimon . 109 `` No Turning Back '' `` The Shocking Moment : Everyone 's Identities Revealed '' `` Shōgeki no toki ! Akasareta tagai no shōtai '' ( 衝撃 の 刻 ! 明か され た 互い の 正体 ) Junichi Sato Sukehiro Tomita Yoshiyuki Shikano Kazuya Kuroda September 24 , 1994 July 7 , 2000 Minako becomes jealous that a Daimon has not yet attacked her , and she fears that she does not have a pure heart . After doing as many `` pure '' things as she can , she is attacked , and her pure heart is taken from her . When Eudial nearly gets away , Usagi is forced to reveal her identity to Haruka & Michiru , who in turn , reveal their identities as Sailors Uranus and Neptune to Sailor Moon . 110 `` Destiny 's Arrival '' `` The Death of Uranus and Neptune : The Talisman Appear '' `` Uranusu - tachi no shi ? Tarisuman shutsugen '' ( ウラヌス 達 の 死 ? タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' `` Seihai no shinpi na chikara ! Mūn nidan henshin '' ( 聖杯 の 神秘 な 力 ! ムーン 二 段 変身 ) Hiroki Shibata Sukehiro Tomita Kazuyuki Hashimoto Ikuko Itō October 22 , 1994 July 11 , 2000 Sailor Pluto appears and reawakens Sailors Uranus and Neptune , and the three talismans resonate with each other ( the third and last being Sailor Pluto 's Garnet Orb ) . The Holy Grail appears , and Sailor Moon grabs it and evolves into Super Sailor Moon , who drives Eudial out . Mimete then kills Eudial by sabotaging her car with snails . 112 `` Show Stoppers '' `` Who is the True Messiah ? Chaos of Light and Darkness '' `` Shin no kyūseishu wa dare ? Hikari to kage no kaosu '' ( 真 の 救世主 は 誰 ? 光 と 影 の カオス ) Noriyo Sasaki Shigeru Yanagawa Minoru Ōkōchi Shinya Hasegawa November 5 , 1994 July 12 , 2000 The girls end up watching the filming of a movie , where Chibiusa encounters Hotaru , whom she quickly befriends . But in the end , she is revealed to be the 12 - year - old daughter of Professor Souichi Tomoe , the possessed leader of the Death Busters . Super Sailor Moon uses her new `` Rainbow Moon Heartache '' attack to defeat the Daimon of the day . 113 `` Rini 's Risky Friendship '' `` A House Filled With Evil Presence : The Beautiful Hotaru 's Secret '' `` Yōki tadayou ie ! Bishōjo Hotaru no himitsu '' ( 妖気 漂う 家 ! 美 少女 ほたる の 秘密 ) Kōnosuke Uda Megumi Sugihara Kenichi Tajiri Taichi Nakamura November 12 , 1994 July 13 , 2000 The three Outer Guardians appear to Tuxedo Mask , and tell him why they can not fight alongside Sailor Moon and the other four Sailor Guardians . Usagi and Chibiusa visit Hotaru , and see a woman who looks like Kaolinite . The Messiah of Silence appears demanding pure hearts . 114 `` Mimet 's Mess '' `` I Love Idols : Mimete 's Dilemma '' `` Aidoru daisuki ! Nayameru Mimetto '' ( アイドル 大好き ! 悩め る ミメット ) Harume Kosaka Yoji Enokido Yoshiyuki Shikano Masahiro Andō November 19 , 1994 July 14 , 2000 Minako and Mimete enter a beauty contest at the same time , with Mimete willing to risk it all to obtain victory . Both fall short of the needed votes to win the idol position , and Mimete decides to just go back to her evil ways . 115 `` The Shadow of Silence '' `` Shadow of Silence : The Pale Glimmer of a Firefly '' `` Chinmoku no kage ! ? Awaki Hotaru hi no yurameki '' ( 沈黙 の 影 ! ? あ わき 蛍火 の ゆらめき ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai November 26 , 1994 July 17 , 2000 Hotaru overexerts herself , and is hospitalized , during which her past without friends is shown , and the reasons for it . Meanwhile , Mimete targets a soap opera actor , during which the attacking Daimon runs near Hotaru , who glows and completely freezes the Daimon . 116 `` Thorny Weather '' `` Sunny Skies After a Storm : A Friendship Dedicated to Hotaru '' `` Arashi nochi hare ! Hotaru ni sasageru yūjō '' ( 嵐 のち 晴れ ! ほたる に 捧げる 友情 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Jouji Yanase December 3 , 1994 July 18 , 2000 Mamoru takes Usagi , Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi , an old classmate of his . While there , Mimete targets Yoshiki 's pure heart and causes a storm that puts the plants at risk . Sailor Moon defeats the accompanying Daimon , U-Bara . 117 `` Heightened Hazard '' `` Higher and Stronger : A Cheer from Usagi '' `` Yori takaku yori tsuyoku ! Usagi no ōen '' ( より 高く より 強く ! うさぎ の 応援 ) Kunihiko Ikuhara Shigeru Yanagawa Minoru Ōkōchi Taichi Nakamura December 10 , 1994 July 19 , 2000 Hotaru wishes to give a letter to Shun Hayase , a famous athlete . He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians ' Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December 17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed '' `` The Messiah of Silence Awakens ? Stars of Destiny '' `` Chinmoku no Meshia kakusei ? Unmei no hoshiboshi '' ( 沈黙 の メシア の 覚せい ? 運命 の 星々 ) Harume Kosaka Yoji Enokido Kazuhisa Asai Shinya Hasegawa December 24 , 1994 September 13 , 2000 The Sailor Guardians visit a planetarium , where a man is attacked by a Daimon . The symbol of Saturn appears on Hotaru 's forehead , and she attacks the Daimon . However , Sailor Uranus , Neptune , and Pluto attack Hotaru , since she is both Sailor Saturn , the Sailor Guardian of Silence , Ruin and Birth , and also possessed by the Messiah of Silence . Super Sailor Moon takes the blow to protect Hotaru . 120 `` Next In Line '' `` An Invasion from an Another Dimension : Mystery of Infinity Academy '' `` Ijigen kara no shinryaku ! Mugen Gakuen no nazo '' ( 異 次元 から の 侵略 ! 無限 学園 の 謎 ) Kōnosuke Uda Megumi Sugihara Minoru Ōkōchi Katsumi Tamegai January 7 , 1995 July 21 , 2000 Chibiusa goes to Hotaru 's house to find that Hotaru has disappeared , and is devastated about it . At the same time , the Sailor Guardians investigate Infinity Academy , where Mimete tries one final time to defeat them before Tellu literally pulls the plug on her operation . 121 `` Fiendish Ferns '' `` A Bewitching Flower That Steals Hearts : Tellu , the Third Witch '' `` Kokoro wo ubau yōka ! Daisan no majō Teruru '' ( 心 を 奪う 妖花 ! 第 三 の 魔女 テルル ) Takuya Igarashi Shigeru Yanagawa Kazuyuki Hashimoto Taichi Nakamura January 14 , 1995 July 24 , 2000 Tellu develops a new plan to steal pure hearts , using a plant bred for that purpose . Setsuna realizes it is an alien plant made to steal pure hearts , and warns the Sailor Guardians . Tuxedo Mask causes a mutated plant to destroy Tellu , leading to them tracing her back to the mysterious Infinity Academy . 122 `` The Science Of Love '' `` Believe in Love : Ami , the Kindhearted Guardian '' `` Ai wo shinjite ! Ami , kokoro yasashiki Senshi '' ( 愛 を 信じ て ! 亜美 心 優しき 戦士 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Hideyuki Motohashi January 21 , 1995 July 25 , 2000 Ami investigates the Infinity Academy , where she meets Viluy , the fourth of the Witches 5 . Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers . But after her battle , her own microscopic robots turn against her , destroying her in the process . 123 `` Wake Up Call '' `` Shadows of Destruction : the Messiah of Silence Awakens '' `` Hametsu no kage ! Chinmoku no Meshia no mezame '' ( 破滅 の 影 ! 沈黙 の メシア の 目覚め ) Harume Kosaka Megumi Sugihara Minoru Ōkōchi Mari Tominaga January 28 , 1995 July 26 , 2000 The last of the Witches 5 , Cyprine steals pure hearts from Infinity Academy students , and the Sailor Guardians engage Cyprine , and her alter - ego , Ptilol , in a showdown . Meanwhile , Chibiusa is abducted by the recently resurrected Kaolinite , and her pure heart is stolen by Mistress 9 , having taken over Hotaru 's body , which allows her to awaken as the Messiah of Silence . 124 `` Who 's Really Who ? '' `` The Coming Terror of Darkness : Struggle of the Eight Guardians '' `` Semari kuru yami no kyōfu ! Kusen no Hassenshi '' ( 迫り 来る 闇 の 恐怖 ! 苦戦 の 8 戦士 ) Takuya Igarashi Yoji Enokido Kazuyuki Hashimoto Ikuko Itō February 4 , 1995 July 28 , 2000 Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy . Sailor Moon is abducted , as she is the holder of the Holy Grail . After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time , Sailor Uranus and Sailor Neptune destroy Germatoid , the Daimon possessing Professor Souichi Tomoe with their talismans , the Deep Aqua Mirror and the Space Sword . 125 `` Darkness , My Old Friend '' `` The Shining Shooting Star : Saturn and the Messiah '' `` Kagayaku ryūsei ! Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei ! Unmei no hoshiboshi betsuri no toki '' ( 新しき 生命 ! 運命 の 星々 別離 の 時 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Taichi Nakamura February 18 , 1995 July 30 , 2000 Haruka and Michiru give the newly reborn infant Hotaru to a still - healing Professor Souichi Tomoe , who is finally free of the Death Busters . However , they viciously attack Sailor Moon , for she protected Hotaru 's body and Sailor Saturn , and risked the whole entire world for one mere life . The Silver Crystal throws them back , and they kneel to her as their leader and leave the city . 127 `` Tough Kindness '' `` A Guardian 's Realization : Strength Lies Within a Pure Heart '' `` Senshi no Jikaku ! Tsuyosa wa Pyua na Kokoro no Naka ni '' ( 戦士 の 自覚 ! 強 さ は 純 な 心 の 中 に ) Harume Kosaka Sukehiro Tomita February 25 , 1995 August 1 , 2000 Chibiusa returns to the future after receiving a letter from her mother , Neo-Queen Serenity . Meanwhile , a final Daimon emerges from the rubble of Infinity Academy , and Chibiusa returns just in time to help defeat it .
  Fact 0.0: The table columns are No., Cloverway Inc. title, original Japanese and Viz Media titles, episode director(s), writer(s), art director(s), animation director(s), original airdate, and English airdate in the United States.
  Fact 0.1: Episode 90 has the Cloverway Inc. title 'Star Struck, Bad Luck', the Viz Media title 'Premonition of the Apocalypse: The Mysterious New Guardians Appear', and the original Japanese title romanized as 'Chikyū hōkai no yokan? Nazo no shin Senshi shutsugen'.
  Fact 0.2: Episode 90 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Masahiro Andō. It originally aired on March 19, 1994 and aired in English in the United States on June 12, 2000.
  Fact 0.3: After experiencing apocalyptic visions, Rei is attacked by a powerful monster called a Daimon.
  Fact 0.4: Sailor Moon is unable to fight the Daimon after it disables her brooch, which reverses her transformation.
  Fact 0.5: The Daimon is defeated by a pair of mysterious new warriors.
  Fact 0.6: The girls despondently realize that a new conflict has begun.
  Fact 0.7: Episode 91 has the Cloverway Inc. title 'Crystal Clear Again', the Viz Media title 'The Rod of Love Is Born: Usagi's New Transformation', and the original Japanese title romanized as 'Ai no roddo tanjō! Usagi no shin henshin'.
  Fact 0.8: Episode 91 was directed by Yūji Endō, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Katsumi Tamegai. It originally aired on March 26, 1994 and aired in English in the United States on June 13, 2000.
  Fact 0.9: While Mamoru strives to cheer Usagi up after her failure in the previous battle, a young girl is the latest target of the Death Busters.
  Fact 0.10: Due to Usagi's inability to transform, the under-powered Sailor Team struggles to fight off the Daimon and is once again forced to rely on the mysterious two warriors.
  Fact 0.11: Once the Sailor Team determines the girl's pure heart is not a talisman, they leave the Sailor Team to fend for themselves.
  Fact 0.12: Tuxedo Mask and Usagi hold hands after the latter's refusal to escape.
  Fact 0.13: Through the power of their love, Usagi's disabled Crystal Star brooch transforms into the Cosmic Heart Compact.
  Fact 0.14: Usagi gains the Spiral Heart Moon Rod and, along with it, the ability to transform and defeat the new enemy.
  Fact 0.15: Episode 92 has the Cloverway Inc. title 'Driving Dangerously', the Viz Media title 'A Handsome Boy? Haruka Tenoh's Secret', and the original Japanese title romanized as 'Suteki na bishōnen? Ten'ō Haruka no himitsu'.
  Fact 0.16: Episode 92 was directed by Kunihiko Ikuhara, written by Katsuyuki Sumisawa, art directed by Kazuhisa Asai, and animation directed by Hisashi Kagawa. It originally aired on April 16, 1994 and aired in English in the United States on June 14, 2000.
  Fact 0.17: Avoiding their study group, Usagi and Minako meet Haruka Tenou, an attractive and supposedly male student of a prestigious local school.
  Fact 0.18: Usagi and Minako both become infatuated with Haruka.
  Fact 0.19: After Minako is defeated in a racing game, Michiru arrives to pick up Haruka.
  Fact 0.20: Michiru and Haruka depart, with Usagi and Minako following behind them soon after.
  Fact 0.21: Once Michiru and Haruka separate at a mechanic's shop, Usagi and Minako take this as their chance.
  Fact 0.22: Haruka tells Usagi and Minako about 'him'-self until a Daimon arrives to attack Kameda, the shop mechanic.
  Fact 0.23: The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down.
  Fact 0.24: Uranus and Neptune arrive to investigate Kameda's pure heart crystal.
  Fact 0.25: When the crystal turns out not to be a talisman, Uranus and Neptune retreat.
  Fact 0.26: Sailor Moon defeats the Daimon.
  Fact 0.27: Later, the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they had told them about.
  Fact 0.28: Michiru arrives with a blonde girl, whom Usagi and Minako initially fail to realize is Haruka.
  Fact 0.29: Once Usagi and Minako realize the blonde girl is Haruka, Minako falls to the ground in anguish.
  Fact 0.30: Haruka notes that she had never said she was male.
  Fact 0.31: Episode 93 has the Cloverway Inc. title 'Bad Harmony', the Viz Media title 'Usagi's Idol: The Graceful Genius Michiru', and the original Japanese title romanized as 'Usagi no akogare! Yūbi no tensai Michiru'.
  Fact 0.32: Episode 93 was directed by Hiroki Shibata, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on April 23, 1994 and aired in English in the United States on June 15, 2000.
  Fact 0.33: When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls, she seeks advice from Michiru Kaioh.
  Fact 0.34: Michiru Kaioh suggests that Usagi and Mamoru go to a violin recital.
  Fact 0.35: Episode 94 has the Cloverway Inc. title 'Swept Off Her Feet', the Viz Media title 'Protect the Pure Heart: The Three-Way Battle', and the original Japanese title romanized as 'Pyua na kokoro wo mamore! Teki mikata mittsu tomoe ransen'.
  Fact 0.36: Episode 94 was directed by Noriyo Sasaki, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on April 30, 1994 and aired in English in the United States on June 16, 2000.
  Fact 0.37: Usagi's friend, Unazuki, is attacked by a Daimon and has her pure heart crystal stolen.
  Fact 0.38: When Sailor Moon tries to take the crystal back to Unazuki, Uranus and Neptune attempt to steal it in order to claim the Talisman.
  Fact 0.39: Episode 95 has the Cloverway Inc. title 'Blinded By Love's Light', the Viz Media title 'Let Moon Help With Your Love Problems', and the original Japanese title romanized as 'Koi no otasuke wa Mūn ni omakase'.
  Fact 0.40: Episode 95 was directed by Harume Kosaka, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on May 7, 1994 and aired in English in the United States on June 19, 2000.
  Fact 0.41: Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other.
  Fact 0.42: Naru and Umino find themselves targeted by a Daimon.
  Fact 0.43: Episode 96 has the Cloverway Inc. title 'Lita Borrows Trouble', the Viz Media title 'Coldhearted Uranus: Makoto in Danger', and the original Japanese title romanized as 'Reikoku na Uranusu? Makoto no pinchi'.
  Fact 0.44: Episode 96 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on May 14, 1994 and aired in English in the United States on June 20, 2000.
  Fact 0.45: Makoto is targeted by a Daimon, but manages to temporarily evade it.
  Fact 0.46: Haruka wants to analyze Makoto's pure heart crystal, and befriends Makoto in order to get close to her.
  Fact 0.47: Episode 97 has the Cloverway Inc. title 'Damp Spirits', the Viz Media title 'The Labyrinth of Water: Ami Targeted', and the original Japanese title romanized as 'Mizu no rabarinsu! Nerawareta Ami'.
  Fact 0.48: Episode 97 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Katsumi Tamegai. It originally aired on May 21, 1994 and aired in English in the United States on June 21, 2000.
  Fact 0.49: Ami's desire to avoid competition leads her to inadvertently offend Michiru in a swimming race.
  Fact 0.50: Usagi accidentally hurts Ami's feelings.
  Fact 0.51: Meanwhile, a Daimon selects Ami as a target.
  Fact 0.52: Episode 98 has the Cloverway Inc. title 'Friendly Foes', the Viz Media title 'To Save Our Friends: Moon and Uranus Join Forces', and the original Japanese title romanized as 'Tomodachi wo sukue! Mūn Uranusu rengō'.
  Fact 0.53: Episode 98 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on May 28, 1994 and aired in English in the United States on June 22, 2000.
  Fact 0.54: After Sailor Neptune is captured and possibly killed by Kaolinite, Sailor Moon and Sailor Uranus find themselves shackled together.
  Fact 0.55: Sailor Moon and Sailor Uranus are forced to reassess their less-than-friendly attitudes toward each other in order to save Sailor Neptune from further harm.
  Fact 0.56: Episode 99 has the Cloverway Inc. title 'Mixed Emotions', the Viz Media title 'A Man's Kindness: Yuichiro Heartbroken by Rei', and the original Japanese title romanized as 'Otoko no yasashisa! Yūichirō Rei ni shitsuren?'.
  Fact 0.57: Episode 99 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Yoshiyuki Shikano, and animation directed by Shinya Hasegawa. It originally aired on June 18, 1994 and aired in English in the United States on June 23, 2000.
  Fact 0.58: Worried by Rei's intensive meditation, Yuuichirou becomes concerned that she has fallen deeply head-over-heels in love for another man.
  Fact 0.59: When Yuuichirou sees Rei with Haruka and mistakes Haruka for a man, he is convinced that 'he' is Rei's new romantic suitor.
  Fact 0.60: Episode 100 has the Cloverway Inc. title 'Individual Happiness', the Viz Media title 'I Want to Quit Being a Sailor Guardian: Minako's Dilemma', and the original Japanese title romanized as 'Sērā Senshi wo yametai!? Minako no nayami'.
  Fact 0.61: Episode 100 was directed by Hiroki Shibata, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Mari Tominaga. It originally aired on June 25, 1994 and aired in English in the United States on June 26, 2000.
  Fact 0.62: Minako is despondent about her double life of being a Sailor Guardian and a normal girl.
  Fact 0.63: Meanwhile, an old friend of Minako's, who she once fell in love with, is attacked by a Daimon.
  Fact 0.64: Minako, as Sailor Venus, must use her skills of volleyball to save the old friend.
  Fact 0.65: Episode 101 has the Cloverway Inc. title 'Birthday Blues, Part 1', the Viz Media title 'Usagi in Tears: a Glass Slipper for My Birthday', and the original Japanese title romanized as 'Usagi namida! Tanjōbi ni garasu no kutsu wo'.
  Fact 0.66: Episode 101 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Kazuyuki Hashimoto, and animation directed by Masahiro Andō. It originally aired on July 2, 1994 and aired in English in the United States on June 27, 2000.
  Fact 0.67: Usagi is furious when Mamoru forgets her 15th birthday.
  Fact 0.68: When Mamoru buys her an expensive pair of glass slippers to make up for it, he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal.
  Fact 0.69: Kaolinite steals Usagi's Cosmic Heart Compact, leaving her unable to transform.
  Fact 0.70: Tuxedo Mask gets captured.
  Fact 0.71: Episode 102 has the Cloverway Inc. title 'Birthday Blues, Part Two', the Viz Media title 'The Stolen Pure Heart: Usagi's Crisis', and the original Japanese title romanized as 'Ubawareta pyua na kokoro! Usagi zettai zetsumei'.
  Fact 0.72: Episode 102 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on July 16, 1994 and aired in English in the United States on June 28, 2000.
  Fact 0.73: With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped, Usagi and the four other Sailor Guardians, along with Sailors Uranus and Neptune, go to retake the compact.
  Fact 0.74: They go to retake the compact so Usagi can become Sailor Moon again and rescue Tuxedo Mask.
  Fact 0.75: They face Kaolinite in a final showdown at Tokyo Tower.
  Fact 0.76: Episode 103 has the Cloverway Inc. title 'Hello, Sailor Mini Moon', the Viz Media title 'The Arrival of a Tiny Pretty Guardian', and the original Japanese title romanized as 'Yatte kita chiccha na Bishōjo Senshi'.
  Fact 0.77: Episode 103 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on August 6, 1994 and aired in English in the United States on June 29, 2000.
  Fact 0.78: Eudial and the Witches 5 are tasked to take over from Kaolinite's position.
  Fact 0.79: The Witches 5 target a genius drummer recruited to play at a festival for Rei.
  Fact 0.80: The Sailor Guardians cannot defeat the Daimon who attacks the drummer.
  Fact 0.81: Chibiusa arrives as the Guardians' newest ally, Sailor Chibi-Moon.
  Fact 0.82: Episode 104 has the Cloverway Inc. title 'Tainted Tea Party', the Viz Media title 'Making New Friends: Chibi Moon's Adventure', and the original Japanese title romanized as 'Tomodachi wo motomete! Chibi Mūn no katsuyaku'.
  Fact 0.83: Episode 104 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Ikuko Itō. It originally aired on August 20, 1994 and aired in English in the United States on June 30, 2000.
  Fact 0.84: Chibiusa has returned from Crystal Tokyo for further training.
  Fact 0.85: Chibiusa attempts to adapt to her new home by making friends her age.
  Fact 0.86: Chibiusa tries to befriend a master who does tea ceremonies, only to become the Death Busters' latest target.
  Fact 0.87: Episode 105 has the Cloverway Inc. title 'People Who Need People', the Viz Media title 'I Want Power: Mako Lost in Doubt', and the original Japanese title romanized as 'Pawā ga hoshī! Mako-chan no mayoi michi'.
  Fact 0.88: Episode 105 was directed by Hiroki Shibata, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on August 27, 1994 and aired in English in the United States on July 3, 2000.
  Fact 0.89: Makoto is training in the mountains, trying to become much stronger after being easily defeated by a Daimon before.
  Fact 0.90: Makoto's trainer is attacked, and tells her that nobody can live alone.
  Fact 0.91: Makoto attacks the Daimon and fails, and realizes she needs her best friends in order to win.
  Fact 0.92: Episode 106 has the Cloverway Inc. title 'Related By Destiny', the Viz Media title 'The Bond of Destiny: Uranus's Distant Past', and the original Japanese title romanized as 'Unmei no kizuna! Uranusu no tōi hi'.
  Fact 0.93: Episode 106 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on September 3, 1994 and aired in English in the United States on July 4, 2000.
  Fact 0.94: Haruka remembers the past when she first met Michiru, who told her that she was destined to become Sailor Uranus.
  Fact 0.95: Haruka initially refused that destiny, for she loved her life too much and was afraid of change.
  Fact 0.96: Haruka was attacked, and Sailor Neptune was injured while defending her.
  Fact 0.97: In her anguish, Haruka took the transformation pen and accepted her destiny as Sailor Uranus, the Sailor Guardian of the Sky.
  Fact 0.98: Episode 107 has the Cloverway Inc. title 'Art Appreciation', the Viz Media title 'Art Is an Explosion of Love: Chibiusa's First Love', and the original Japanese title romanized as 'Geijutsu wa ai no bakuhatsu! Chibiusa no hatsukoi'.
  Fact 0.99: Episode 107 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on September 10, 1994 and aired in English in the United States on July 5, 2000.
  Fact 0.100: Chibiusa meets a boy named Masanori at her art class and develops a crush on him.
  Fact 0.101: Masanori already has a crush on Michiru, who is taking a class at that school as well.
  Fact 0.102: When a Daimon attacks Masanori, Chibiusa must take a stand to protect him.
  Fact 0.103: Episode 108 has the Cloverway Inc. title 'Everything's Coming Up Rosey', the Viz Media title 'Usagi Dancing to the Waltz', and the original Japanese title romanized as 'Usagi no dansu wa warutsu ni notte'.
  Fact 0.104: Episode 108 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on September 17, 1994 and aired in English in the United States on July 6, 2000.
  Fact 0.105: An English gentleman named Edwards invites Mamoru and the girls to a fancy party.
  Fact 0.106: At the party, Usagi accidentally becomes drunk.
  Fact 0.107: At the party, the team is forced to confront a fashionable Eudial and her latest Daimon.
  Fact 0.108: Episode 109 has the Cloverway Inc. title 'No Turning Back', the Viz Media title 'The Shocking Moment: Everyone's Identities Revealed', and the original Japanese title romanized as 'Shōgeki no toki! Akasareta tagai no shōtai'.
  Fact 0.109: Episode 109 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on September 24, 1994 and aired in English in the United States on July 7, 2000.
  Fact 0.110: Minako becomes jealous that a Daimon has not yet attacked her, and she fears that she does not have a pure heart.
  Fact 0.111: After doing as many 'pure' things as she can, Minako is attacked, and her pure heart is taken from her.
  Fact 0.112: When Eudial nearly gets away, Usagi is forced to reveal her identity to Haruka and Michiru.
  Fact 0.113: Haruka and Michiru reveal their identities as Sailors Uranus and Neptune to Sailor Moon.
  Fact 0.114: Episode 110 has the Cloverway Inc. title 'Destiny's Arrival', the Viz Media title 'The Death of Uranus and Neptune: The Talisman Appear', and the original Japanese title romanized as 'Uranusu-tachi no shi? Tarisuman shutsugen'.
  Fact 0.115: Episode 110 was directed by Kunihiko Ikuhara, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Mari Tominaga. It originally aired on October 15, 1994 and aired in English in the United States on July 10, 2000.
  Fact 0.116: Faced with losing her position at Witches 5, Eudial creates a computer program to find the true owners of the three Talismans.
  Fact 0.117: Eudial learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts.
  Fact 0.118: Eudial and her companions take away Usagi's brooch, and follow Eudial onto a lonely island.
  Fact 0.119: The invitation turns out to be a trick, and Michiru's pure heart is taken, which turns into the mystical Deep Aqua Mirror.
  Fact 0.120: Haruka, in order to release her own talisman, shoots herself with Eudial's gun.
  Fact 0.121: Haruka's pure heart transforms into the mystical Space Sword.
  Fact 0.122: Episode 111 has the Cloverway Inc. title 'The Purity Chalice', the Viz Media title 'The Holy Grail's Mystical Power: Moon's Double Transformation', and the original Japanese title romanized as 'Seihai no shinpi na chikara! Mūn nidan henshin'.
  Fact 0.123: Episode 111 was directed by Hiroki Shibata, written by Sukehiro Tomita, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on October 22, 1994 and aired in English in the United States on July 11, 2000.
  Fact 0.124: Sailor Pluto appears and reawakens Sailors Uranus and Neptune.
  Fact 0.125: The three talismans resonate with each other, the third and last being Sailor Pluto's Garnet Orb.
  Fact 0.126: The Holy Grail appears, and Sailor Moon grabs it and evolves into Super Sailor Moon.
  Fact 0.127: Super Sailor Moon drives Eudial out.
  Fact 0.128: Mimete kills Eudial by sabotaging her car with snails.
  Fact 0.129: Episode 112 has the Cloverway Inc. title 'Show Stoppers', the Viz Media title 'Who is the True Messiah? Chaos of Light and Darkness', and the original Japanese title romanized as 'Shin no kyūseishu wa dare? Hikari to kage no kaosu'.
  Fact 0.130: Episode 112 was directed by Noriyo Sasaki, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Shinya Hasegawa. It originally aired on November 5, 1994 and aired in English in the United States on July 12, 2000.
  Fact 0.131: The girls end up watching the filming of a movie, where Chibiusa encounters Hotaru, whom she quickly befriends.
  Fact 0.132: In the end, Chibiusa is revealed to be the 12-year-old daughter of Professor Souichi Tomoe, the possessed leader of the Death Busters.
  Fact 0.133: Super Sailor Moon uses her new 'Rainbow Moon Heartache' attack to defeat the Daimon of the day.
  Fact 0.134: Episode 113 has the Cloverway Inc. title 'Rini's Risky Friendship', the Viz Media title 'A House Filled With Evil Presence: The Beautiful Hotaru's Secret', and the original Japanese title romanized as 'Yōki tadayou ie! Bishōjo Hotaru no himitsu'.
  Fact 0.135: Episode 113 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on November 12, 1994 and aired in English in the United States on July 13, 2000.
  Fact 0.136: The three Outer Guardians appear to Tuxedo Mask, and tell him why they cannot fight alongside Sailor Moon and the other four Sailor Guardians.
  Fact 0.137: Usagi and Chibiusa visit Hotaru, and see a woman who looks like Kaolinite.
  Fact 0.138: The Messiah of Silence appears demanding pure hearts.
  Fact 0.139: Episode 114 has the Cloverway Inc. title 'Mimet's Mess', the Viz Media title 'I Love Idols: Mimete's Dilemma', and the original Japanese title romanized as 'Aidoru daisuki! Nayameru Mimetto'.
  Fact 0.140: Episode 114 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Yoshiyuki Shikano, and animation directed by Masahiro Andō. It originally aired on November 19, 1994 and aired in English in the United States on July 14, 2000.
  Fact 0.141: Minako and Mimete enter a beauty contest at the same time, with Mimete willing to risk it all to obtain victory.
  Fact 0.142: Both Minako and Mimete fall short of the needed votes to win the idol position.
  Fact 0.143: Mimete decides to go back to her evil ways.
  Fact 0.144: Episode 115 has the Cloverway Inc. title 'The Shadow of Silence', the Viz Media title 'Shadow of Silence: The Pale Glimmer of a Firefly', and the original Japanese title romanized as 'Chinmoku no kage!? Awaki Hotaru hi no yurameki'.
  Fact 0.145: Episode 115 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on November 26, 1994 and aired in English in the United States on July 17, 2000.
  Fact 0.146: Hotaru overexerts herself and is hospitalized, during which her past without friends is shown, and the reasons for it.
  Fact 0.147: Meanwhile, Mimete targets a soap opera actor.
  Fact 0.148: During the attack on the actor, the attacking Daimon runs near Hotaru, who glows and completely freezes the Daimon.
  Fact 0.149: Episode 116 has the Cloverway Inc. title 'Thorny Weather', the Viz Media title 'Sunny Skies After a Storm: A Friendship Dedicated to Hotaru', and the original Japanese title romanized as 'Arashi nochi hare! Hotaru ni sasageru yūjō'.
  Fact 0.150: Episode 116 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Jouji Yanase. It originally aired on December 3, 1994 and aired in English in the United States on July 18, 2000.
  Fact 0.151: Mamoru takes Usagi, Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi, an old classmate of Mamoru's.
  Fact 0.152: While there, Mimete targets Yoshiki's pure heart and causes a storm that puts the plants at risk.
  Fact 0.153: Sailor Moon defeats the accompanying Daimon, U-Bara.
  Fact 0.154: Episode 117 has the Cloverway Inc. title 'Heightened Hazard', the Viz Media title 'Higher and Stronger: A Cheer from Usagi', and the original Japanese title romanized as 'Yori takaku yori tsuyoku! Usagi no ōen'.
  Fact 0.155: Episode 117 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on December 10, 1994 and aired in English in the United States on July 19, 2000.
  Fact 0.156: Hotaru wishes to give a letter to Shun Hayase, a famous athlete.
  Fact 0.157: Shun Hayase becomes targeted when Hotaru goes to give the letter to him.
  Fact 0.158: Hotaru possessively blasts the enemy with a hugely strong amount of energy.
  Fact 0.159: Hotaru talks to Shun about being frail and weak, and he thanks her for talking to him.
  Fact 0.160: Episode 118 has the Cloverway Inc. title 'It's In The Cards', the Viz Media title 'The Battle Inside the Demonic Space: The Sailor Guardians' Gamble', and the original Japanese title romanized as 'Makū no tatakai! Sērā Senshi no kake'.
  Fact 0.161: Episode 118 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Masahide Yanagisawa. It originally aired on December 17, 1994 and aired in English in the United States on July 20, 2000.
  Fact 0.162: Mimete attempts to create a Daimon by herself, throwing the entire Tomoe house into another dimension.
  Fact 0.163: The Guardians enter the dimension and are defeated by a game-master Daimon named U-Ikasaman.
  Fact 0.164: Hotaru manages to assist in U-Ikasaman's defeat by the use of a bluff.
  Fact 0.165: Episode 119 has the Cloverway Inc. title 'Goodness Eclipsed', the Viz Media title 'The Messiah of Silence Awakens? Stars of Destiny', and the original Japanese title romanized as 'Chinmoku no Meshia kakusei? Unmei no hoshiboshi'.
  Fact 0.166: Episode 119 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Shinya Hasegawa. It originally aired on December 24, 1994 and aired in English in the United States on September 13, 2000.
  Fact 0.167: The Sailor Guardians visit a planetarium, where a man is attacked by a Daimon.
  Fact 0.168: The symbol of Saturn appears on Hotaru's forehead, and she attacks the Daimon.
  Fact 0.169: Sailor Uranus, Neptune, and Pluto attack Hotaru, since she is both Sailor Saturn, the Sailor Guardian of Silence, Ruin and Birth, and also possessed by the Messiah of Silence.
  Fact 0.170: Super Sailor Moon takes the blow to protect Hotaru.
  Fact 0.171: Episode 120 has the Cloverway Inc. title 'Next In Line', the Viz Media title 'An Invasion from an Another Dimension: Mystery of Infinity Academy', and the original Japanese title romanized as 'Ijigen kara no shinryaku! Mugen Gakuen no nazo'.
  Fact 0.172: Episode 120 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Katsumi Tamegai. It originally aired on January 7, 1995 and aired in English in the United States on July 21, 2000.
  Fact 0.173: Chibiusa goes to Hotaru's house to find that Hotaru has disappeared, and is devastated about it.
  Fact 0.174: At the same time, the Sailor Guardians investigate Infinity Academy.
  Fact 0.175: Mimete tries one final time to defeat the Sailor Guardians before Tellu literally pulls the plug on her operation.
  Fact 0.176: Episode 121 has the Cloverway Inc. title 'Fiendish Ferns', the Viz Media title 'A Bewitching Flower That Steals Hearts: Tellu, the Third Witch', and the original Japanese title romanized as 'Kokoro wo ubau yōka! Daisan no majō Teruru'.
  Fact 0.177: Episode 121 was directed by Takuya Igarashi, written by Shigeru Yanagawa, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on January 14, 1995 and aired in English in the United States on July 24, 2000.
  Fact 0.178: Tellu develops a new plan to steal pure hearts, using a plant bred for that purpose.
  Fact 0.179: Setsuna realizes the plant is an alien plant made to steal pure hearts, and warns the Sailor Guardians.
  Fact 0.180: Tuxedo Mask causes a mutated plant to destroy Tellu.
  Fact 0.181: This leads the Sailor Guardians to trace Tellu back to the mysterious Infinity Academy.
  Fact 0.182: Episode 122 has the Cloverway Inc. title 'The Science Of Love', the Viz Media title 'Believe in Love: Ami, the Kindhearted Guardian', and the original Japanese title romanized as 'Ai wo shinjite! Ami, kokoro yasashiki Senshi'.
  Fact 0.183: Episode 122 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Hideyuki Motohashi. It originally aired on January 21, 1995 and aired in English in the United States on July 25, 2000.
  Fact 0.184: Ami investigates the Infinity Academy, where she meets Viluy, the fourth of the Witches 5.
  Fact 0.185: Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers.
  Fact 0.186: After Viluy's battle, her own microscopic robots turn against her, destroying her in the process.
  Fact 0.187: Episode 123 has the Cloverway Inc. title 'Wake Up Call', the Viz Media title 'Shadows of Destruction: the Messiah of Silence Awakens', and the original Japanese title romanized as 'Hametsu no kage! Chinmoku no Meshia no mezame'.
  Fact 0.188: Episode 123 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Mari Tominaga. It originally aired on January 28, 1995 and aired in English in the United States on July 26, 2000.
  Fact 0.189: Cyprine is the last of the Witches 5.
  Fact 0.190: Cyprine steals pure hearts from Infinity Academy students.
  Fact 0.191: The Sailor Guardians engage Cyprine, and her alter-ego, Ptilol, in a showdown.
  Fact 0.192: Chibiusa is abducted by the recently resurrected Kaolinite.
  Fact 0.193: Chibiusa's pure heart is stolen by Mistress 9, who has taken over Hotaru's body, which allows her to awaken as the Messiah of Silence.
  Fact 0.194: Episode 124 has the Cloverway Inc. title 'Who's Really Who?', the Viz Media title 'The Coming Terror of Darkness: Struggle of the Eight Guardians', and the original Japanese title romanized as 'Semari kuru yami no kyōfu! Kusen no Hassenshi'.
  Fact 0.195: Episode 124 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on February 4, 1995 and aired in English in the United States on July 28, 2000.
  Fact 0.196: Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy.
  Fact 0.197: Sailor Moon is abducted, as she is the holder of the Holy Grail.
  Fact 0.198: After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time, Sailor Uranus and Sailor Neptune destroy Germatoid.
  Fact 0.199: Germatoid is the Daimon possessing Professor Souichi Tomoe.
  Fact 0.200: Sailor Uranus and Sailor Neptune destroy Germatoid with their talismans, the Deep Aqua Mirror and the Space Sword.
  Fact 0.201: Episode 125 has the Cloverway Inc. title 'Darkness, My Old Friend', the Viz Media title 'The Shining Shooting Star: Saturn and the Messiah', and the original Japanese title romanized as 'Kagayaku ryūsei! Satān soshite kyūseishu'.
  Fact 0.202: Episode 125 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on February 11, 1995 and aired in English in the United States on July 29, 2000.
  Fact 0.203: Sailor Moon is tricked into giving Mistress 9 the Holy Grail, which she uses to summon Pharaoh 90 to Earth.
  Fact 0.204: Hotaru regains consciousness with the assistance of Sailor Moon and her father, destroying Mistress 9.
  Fact 0.205: Hotaru fully awakens the spirit of Sailor Saturn within herself at last.
  Fact 0.206: Sailor Saturn returns Chibiusa's pure heart crystal.
  Fact 0.207: Sailor Saturn appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers.
  Fact 0.208: Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more.
  Fact 0.209: Super Sailor Moon saves both Hotaru and Sailor Saturn.
  Fact 0.210: Sailor Saturn is reborn as a baby immediately afterward.
  Fact 0.211: Episode 126 has the Cloverway Inc. title 'Second Chance', the Viz Media title 'A New Life: Parting of the Stars of Destiny', and the original Japanese title romanized as 'Atarashiki seimei! Unmei no hoshiboshi betsuri no toki'.
  Fact 0.212: Episode 126 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Taichi Nakamura. It originally aired on February 18, 1995 and aired in English in the United States on July 30, 2000.
  Fact 0.213: Haruka and Michiru give the newly reborn infant Hotaru to a still-healing Professor Souichi Tomoe, who is finally free of the Death Busters.
  Fact 0.214: Haruka and Michiru viciously attack Sailor Moon, for she protected Hotaru's body and Sailor Saturn, and risked the whole entire world for one mere life.
  Fact 0.215: The Silver Crystal throws Haruka and Michiru back.
  Fact 0.216: Haruka and Michiru kneel to Sailor Moon as their leader and leave the city.
  Fact 0.217: Episode 127 has the Cloverway Inc. title 'Tough Kindness', the Viz Media title 'A Guardian's Realization: Strength Lies Within a Pure Heart', and the original Japanese title romanized as 'Senshi no Jikaku! Tsuyosa wa Pyua na Kokoro no Naka ni'.
  Fact 0.218: Episode 127 was directed by Harume Kosaka and written by Sukehiro Tomita. It originally aired on February 25, 1995 and aired in English in the United States on August 1, 2000.
  Fact 0.219: Chibiusa returns to the future after receiving a letter from her mother, Neo-Queen Serenity.
  Fact 0.220: Meanwhile, a final Daimon emerges from the rubble of Infinity Academy.
  Fact 0.221: Chibiusa returns just in time to help defeat the final Daimon.

Retrieved items:
<item 1>
List of Sailor Moon S episodes
Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers . But after her battle , her own microscopic robots turn against her ,
destroying her in the process . 123 `` Wake Up Call '' `` Shadows of Destruction : the Messiah of Silence Awakens '' `` Hametsu no kage ! Chinmoku no Meshia no
mezame '' ( 破滅 の 影 ! 沈黙 の メシア の 目覚め ) Harume Kosaka Megumi Sugihara Minoru Ōkōchi Mari Tominaga January 28 , 1995 July 26 , 2000 The last of the Witches 5 ,
Cyprine steals pure hearts from Infinity Academy students , and the Sailor Guardians engage Cyprine , and her alter - ego , Ptilol , in a showdown . Meanwhile ,
Chibiusa is abducted by the recently resurrected Kaolinite , and her pure heart is stolen by Mistress 9 , having taken over Hotaru 's body , which allows her to
awaken as the Messiah of Silence . 124 `` Who 's Really Who ? ''
</item 1>

<item 2>
List of Sailor Moon S episodes
Mūn nidan henshin '' ( 聖杯 の 神秘 な 力 ! ムーン 二 段 変身 ) Hiroki Shibata Sukehiro Tomita Kazuyuki Hashimoto Ikuko Itō October 22 , 1994 July 11 , 2000 Sailor Pluto
appears and reawakens Sailors Uranus and Neptune , and the three talismans resonate with each other ( the third and last being Sailor Pluto 's Garnet Orb ) .
The Holy Grail appears , and Sailor Moon grabs it and evolves into Super Sailor Moon , who drives Eudial out . Mimete then kills Eudial by sabotaging her car
with snails . 112 `` Show Stoppers '' `` Who is the True Messiah ? Chaos of Light and Darkness '' `` Shin no kyūseishu wa dare ? Hikari to kage no kaosu '' ( 真
の 救世主 は 誰 ?
</item 2>

<item 3>
List of Sailor Moon S episodes
After Minako is defeated in a racing game , Michiru arrives to pick up Haruka , and the two depart with Usagi and Minako soon following behind them . Once
Michiru and Haruka separate at a mechanic 's shop , Usagi and Minako take this as their chance , and Haruka tells them about `` him '' - self until a Daimon
arrives to attack Kameda , the shop mechanic . The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down , when
Uranus and Neptune arrive to investigate Kameda 's pure heart crystal . When it turns out not to be a talisman they retreat and Sailor Moon defeats the Daimon .
Later , the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they 'd told them about ; Michiru arrives with a
blonde girl , whom Usagi and Minako initially fail to realize is Haruka ; once they do , Minako falls to the ground in anguish , and Haruka notes that she 'd
never said she was male .
</item 3>

<item 4>
List of Sailor Moon S episodes
Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked
into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her
father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart
crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor
Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby
immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei !
</item 4>

<item 5>
List of Sailor Moon S episodes
タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial
creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within
their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's
pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her
pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' ``
Seihai no shinpi na chikara !
</item 5>

<item 6>
List of Sailor Moon S episodes
Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch
transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92
`` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko
Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an
attractive and supposedly male student of a prestigious local school , with whom they both become infatuated .
</item 6>

<item 7>
List of Sailor Moon S episodes
124 `` Who 's Really Who ? '' `` The Coming Terror of Darkness : Struggle of the Eight Guardians '' `` Semari kuru yami no kyōfu ! Kusen no Hassenshi '' ( 迫り 来る
闇 の 恐怖 ! 苦戦 の 8 戦士 ) Takuya Igarashi Yoji Enokido Kazuyuki Hashimoto Ikuko Itō February 4 , 1995 July 28 , 2000 Mistress 9 uses all remaining Daimon eggs to
create a powerful energized shield of evil minions around Infinity Academy . Sailor Moon is abducted , as she is the holder of the Holy Grail . After witnessing
Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time , Sailor Uranus and Sailor Neptune destroy Germatoid , the
Daimon possessing Professor Souichi Tomoe with their talismans , the Deep Aqua Mirror and the Space Sword . 125 `` Darkness , My Old Friend '' `` The Shining
Shooting Star : Saturn and the Messiah '' `` Kagayaku ryūsei !
</item 7>

<item 8>
List of Sailor Moon S episodes
明か され た 互い の 正体 ) Junichi Sato Sukehiro Tomita Yoshiyuki Shikano Kazuya Kuroda September 24 , 1994 July 7 , 2000 Minako becomes jealous that a Daimon has not
yet attacked her , and she fears that she does not have a pure heart . After doing as many `` pure '' things as she can , she is attacked , and her pure heart
is taken from her . When Eudial nearly gets away , Usagi is forced to reveal her identity to Haruka & Michiru , who in turn , reveal their identities as Sailors
Uranus and Neptune to Sailor Moon . 110 `` Destiny 's Arrival '' `` The Death of Uranus and Neptune : The Talisman Appear '' `` Uranusu - tachi no shi ?
Tarisuman shutsugen '' ( ウラヌス 達 の 死 ?
</item 8>

<item 9>
List of Sailor Moon S episodes
He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about
being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians '
Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December
17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the
dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed
'' `` The Messiah of Silence Awakens ?
</item 9>

<item 10>
List of Sailor Moon S episodes
The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear
Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō
Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous
battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight
off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the
Sailor Team to fend for themselves .
</item 10>
=== END ITEM id=5367a2a349f0a8ef ===
