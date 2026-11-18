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

=== ITEM id=4c5c41e3a6adf648 ===
Gold sentences and facts:
Sentence 0: Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse , which can lead to twin babies from two separate biological fathers . The term superfecundation is derived from fecund , meaning the ability to produce offspring . Heteropaternal superfecundation refers to the fertilization of two separate ova by two different fathers . Homopaternal superfecundation refers to the fertilization of two separate ova from the same father , leading to fraternal twins . While heteropaternal superfecundation is referred to as a form of atypical twinning , genetically , the twins are half siblings . Superfecundation , while rare , can occur through either separate occurrences of sexual intercourse or through artificial insemination .
  Fact 0.0: Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse.
  Fact 0.1: Superfecundation can lead to twin babies from two separate biological fathers.
  Fact 0.2: The term superfecundation is derived from fecund.
  Fact 0.3: Fecund means the ability to produce offspring.
  Fact 0.4: Heteropaternal superfecundation refers to the fertilization of two separate ova by two different fathers.
  Fact 0.5: Homopaternal superfecundation refers to the fertilization of two separate ova from the same father.
  Fact 0.6: Homopaternal superfecundation leads to fraternal twins.
  Fact 0.7: Heteropaternal superfecundation is referred to as a form of atypical twinning.
  Fact 0.8: Genetically, the twins from heteropaternal superfecundation are half siblings.
  Fact 0.9: Superfecundation is rare.
  Fact 0.10: Superfecundation can occur through separate occurrences of sexual intercourse.
  Fact 0.11: Superfecundation can occur through artificial insemination.

Retrieved items:
<item 1>
Superfecundation
This is known as superfetation . Heteropaternal superfecundation in mammals ( edit ) Heteropaternal superfecundation is common in animals such as cats and dogs
. Stray dogs can produce litters in which every puppy has a different sire . Though rare in humans , cases have been documented . In one study on humans , the
frequency was 2.4 % among dizygotic twins whose parents had been involved in paternity suits . Cases involving superfecundation ( edit ) Koen and Teun Stuart ,
Dutch boys who were the result of in vitro fertilization ( IVF ) . In a mixup at the laboratory , equipment had been used twice , causing another man 's sperm
to be mixed with the intended father 's . In 1995 , a young woman gave birth to diamniotic monochorionic twins , who were originally assumed to be monozygotic
twins until a paternity suit led to a DNA test . This led to the discovery that the twins had different fathers .
</item 1>

<item 2>
Superfecundation
Contents ( hide ) 1 Conception 2 Heteropaternal superfecundation in mammals 3 Cases involving superfecundation 4 See also 5 References 6 Further reading
Conception ( edit ) Superfecundation most commonly happens within hours or days of the first instance of fertilization with ova released during the same cycle .
The time window when eggs are able to be fertilized is small . Sperm cells can live inside a female 's body for four to five days . Once ovulation occurs , the
egg remains viable for 12 -- 48 hours before it begins to disintegrate . Thus , the fertile period can span five to seven days . Ovulation is usually suspended
during pregnancy to prevent further ova becoming fertilized and to help increase the chances of a full - term pregnancy . However , if an ovum is released after
the female was already impregnated when previously ovulating , a chance of a second pregnancy occurs , albeit at a different stage of development . This is
known as superfetation .
</item 2>

<item 3>
Superfecundation
This led to the discovery that the twins had different fathers . In 1982 , twins who were born with two different skin colors were discovered to be conceived as
a result of heteropaternal superfecundation . In 2001 , a case of spontaneous monopaternal superfecundation was reported after a woman undergoing IVF treatments
gave birth to quintuplets after only two embryos were implanted . Genetic testing supported that the twinning was not a result of the embryos splitting , and
that all five boys shared the same father . In 2016 , an IVF - implanted surrogate mother gave birth to two children : one genetically unrelated child from an
implanted embryo , and a biological child from her own egg and her husband 's sperm . See also ( edit ) Castor and Pollux , said to be twins having different
fathers Chimera ( genetics ) Mixed twins Polyspermy References ( edit ) Jump up ^ Worland , Justin ( May 8 , 2015 ) . `` The Science of How Women Can Have Twins
With 2 Different Fathers '' . Time . Retrieved March 23 , 2017 .
</item 3>

<item 4>
Superfecundation
PMID 1488855 . Jump up ^ Fierro , Pamela Prindle ( January 5 , 2016 ) , `` Heteropaternal Superfecundation - Twins with Different Fathers '' , About.com ,
archived from the original on April 4 , 2016 Jump up ^
https://www.scopus.com/record/display.uri?eid=2-s2.0-0020008620&origin=inward&txGid=F6227767DB3CDB724EA92EFC60A5A985.wsnAw8kcdt7IPYLO0V48gA%3a2 Jump up ^
Amsalem , Hagai ; Tsvieli , Rimona ; Zentner , Bat Sheva ; Yagel , Simcha ; Mitrani - Rosenbaum , Stella ; Hurwitz , Arye ( 2001 ) . `` Monopaternal
superfecundation of quintuplets after transfer of two embryos in an in vitro fertilization cycle '' . Fertility and Sterility. 76 ( 3 ) : 621 -- 3 .
</item 4>

<item 5>
Human fertilization
This possibility has been investigated by computer simulations of the fertilization process . See also ( edit ) Spontaneous conception , the unassisted
conception of a subsequent child after prior use of assisted reproductive technology References ( edit ) Jump up ^ Garrison , Fielding . An Introduction to the
History of Medicine , pages 566 - 567 ( Saunders 1921 ) . Jump up ^ `` Archived copy '' . Archived from the original on 2011 - 12 - 22 . Retrieved 2016 - 01 -
24 . Jump up ^ http://www.americanpregnancy.org/preventingpregnancy/pregnancyfaqmyths.html Jump up ^ Lawyers Guide to Forensic Medicine SBN 978 - 1 - 85941 -
159 - 9 By Bernard Knight - Page 188 `` Pregnancy is well known to occur from such external ejaculation ... '' Jump up ^ `` Fertilization of the Ovum '' . Gray
's Anatomy . Retrieved 2010 - 10 - 16 . Jump up ^ `` Fertilization '' . Retrieved 28 July 2010 .
</item 5>

<item 6>
Superfecundation
PMID 8064269 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Superfecundation&oldid=818981440 '' Categories : Reproduction Twin Triplets Multiple
births Free sex Sibling Hidden categories : Articles needing more detailed references All articles with dead external links Articles with dead external links
from June 2017 Articles needing additional references from May 2009 All articles needing additional references Talk Contents About Wikipedia Deutsch فارسی
Français Nederlands Polski Português Русский Slovenčina Edit links This page was last edited on 6 January 2018 , at 18 : 28 . About Wikipedia
</item 6>

<item 7>
Superfecundation
Superfecundation - wikipedia Superfecundation Jump to : navigation , search This article needs additional citations for verification . Please help improve this
article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2009 ) ( Learn how and when to remove this template
message ) Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse , which can lead to
twin babies from two separate biological fathers . The term superfecundation is derived from fecund , meaning the ability to produce offspring . Heteropaternal
superfecundation refers to the fertilization of two separate ova by two different fathers . Homopaternal superfecundation refers to the fertilization of two
separate ova from the same father , leading to fraternal twins . While heteropaternal superfecundation is referred to as a form of atypical twinning ,
genetically , the twins are half siblings . Superfecundation , while rare , can occur through either separate occurrences of sexual intercourse or through
artificial insemination .
</item 7>

<item 8>
Superfecundation
Fertility and Sterility. 76 ( 3 ) : 621 -- 3 . doi : 10.1016 / S0015 - 0282 ( 01 ) 01976 - 8 . PMID 11532493 . Jump up ^ Susan Scutti ( November 3 , 2017 ) . ``
Surrogate mom delivers two babies , one her own '' . CNN . Further reading ( edit ) `` Doctors Discover Dallas Twins Have Two Different Dads '' , Fox News , Sky
News , 2009 Lu , H.L. ; Wang , C.X. ; Wu , F.Q. ; Li , J.J. ( 1994 ) . `` Paternity identification in twins with different fathers '' . Journal of forensic
sciences . 39 ( 4 ) : 1100 -- 2 . PMID 8064269 .
</item 8>

<item 9>
Superfecundation
Time . Retrieved March 23 , 2017 . Jump up ^ `` Public Education Pamphlets '' . sogc.org . ^ Jump up to : Ambach , E ; Parson , W ; Brezinka , C ( 2000 ) . ``
Superfecundation and dual paternity in a twin pregnancy ending with placental abruption '' . Journal of forensic sciences . 45 ( 1 ) : 181 -- 3 . PMID 10641935
. ^ Jump up to : McNamara , Helen C. ; Kane , Stefan C. ; Craig , Jeffrey M. ; Short , Roger V. ; Umstad , Mark P. ( 2016 ) . `` A review of the mechanisms and
evidence for typical and atypical twinning '' . American Journal of Obstetrics and Gynecology. 214 ( 2 ) : 172 -- 91 . doi : 10.1016 / j. ajog. 2015.10. 930 .
PMID 26548710 . Jump up ^ Wenk , R.E.
</item 9>

<item 10>
Superfecundation
PMID 26548710 . Jump up ^ Wenk , R.E. ; Houtz , T ; Brooks , M ; Chiafari , F.A. ( 1992 ) . `` How frequent is heteropaternal superfecundation ? '' . Acta
geneticae medicae et gemellologiae. 41 ( 1 ) : 43 -- 7 . PMID 1488855 .
</item 10>
=== END ITEM id=4c5c41e3a6adf648 ===

=== ITEM id=1bffe1e8651e8150 ===
Gold sentences and facts:
Sentence 0: The Battle of Lake Erie , sometimes called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval battles of the War of 1812 .
  Fact 0.0: The Battle of Lake Erie is sometimes called the Battle of Put-in-Bay.
  Fact 0.1: The Battle of Lake Erie was fought on 10 September 1813.
  Fact 0.2: The Battle of Lake Erie was fought in Lake Erie off the coast of Ohio.
  Fact 0.3: The Battle of Lake Erie was fought during the War of 1812.
  Fact 0.4: Nine vessels of the United States Navy defeated six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.5: Nine vessels of the United States Navy captured six vessels of the British Royal Navy in the Battle of Lake Erie.
  Fact 0.6: The Battle of Lake Erie ensured American control of the lake for the rest of the War of 1812.
  Fact 0.7: The Battle of Lake Erie allowed the Americans to recover Detroit.
  Fact 0.8: The Battle of Lake Erie allowed the Americans to win the Battle of the Thames.
  Fact 0.9: Winning the Battle of the Thames broke the Indian confederation of Tecumseh.
  Fact 0.10: The Battle of Lake Erie was one of the biggest naval battles of the War of 1812.

Retrieved items:
<item 1>
At the war’s outset, Britain controlled Lake Erie; its capture of Detroit and the American vessel Adams helped defeat General Hull at Detroit. In October 1812,
Americans captured the British brig Caledonia and Detroit; Detroit was burned, while Caledonia joined the U.S. Navy. After the 1813 Battle of Lake Erie, Procter
retreated, and Harrison defeated him at the Thames, where Tecumseh was killed. U.S. lake control enabled later successes and protected the
</item 1>

<item 2>
Battle of Lake Erie
Battle of Lake Erie - wikipedia Battle of Lake Erie Jump to : navigation , search Battle of Lake Erie Part of the War of 1812 Battle of Lake Erie by William
Henry Powell , painted 1865 , shows Oliver Hazard Perry transferring from Lawrence to Niagara Date 10 September 1813 Location Lake Erie , near Put - in - Bay ,
Ohio Result Decisive American victory Belligerents United States United Kingdom Commanders and leaders Oliver Hazard Perry Jesse Elliot Robert Heriot Barclay
Robert Finnis Strength 5 schooners 3 brigs 1 sloop 2 ships 1 brig 2 schooners 1 sloop Casualties and losses 27 killed 96 wounded 1 brig severely damaged 41
killed 93 wounded 306 captured entire squadron captured Detroit frontier Tippecanoe Fort Mackinac ( 1812 ) Brownstown Maguaga Fort Dearborn Detroit Fort
Harrison Fort Wayne Wild Cat Creek Mississinewa Frenchtown Africa Point Fort Meigs Fort Stephenson Lake Erie Thames Longwoods Prairie du Chien Campbell Island
Mackinac Island ( 1814 ) Lake Huron Malcolm 's Mills Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS
</item 2>

<item 3>
Battle of Lake Erie
Michaels Raid on Havre de Grace Massachusetts Battle of Hampden New York Battle of Big Sandy Creek Battle of Buffalo Battle of Ogdensburg Battle of Plattsburgh
Capture of Fort Niagara Raid on Black Rock Second Battle of Sacket 's Harbor Ohio Battle of Fort Stephenson Copus massacre Siege of Fort Meigs Virginia Battle
of Craney Island Raid on Alexandria Skirmish at Farnham Church Battle of Rappahannock River U.S. territories Alabama Battles of Fort Bowyer Illinois Battle of
Fort Dearborn Battle of Rock Island Rapids Siege of Prairie du Chien Indiana Battle of the Mississinewa Battle of Tippecanoe Battle of Wild Cat Creek Siege of
Fort Harrison Siege of Fort Wayne Michigan Battle of Brownstown Battle of Frenchtown Battle of Mackinac Island Battle of Maguaga Siege of Detroit Siege of Fort
Mackinac Mississippi Battle of Burnt Corn Battle of Callabee Creek Canoe Fight Battle of Holy Ground Battle of Horseshoe Bend Battle of Talladega Battle of
Tallushatchee Battles of Emuckfaw and Enotachopo Creek Fort Mims massacre Kimbell -- James Massacre Missouri Battle of Credit Island Battle of the Sink Hole
British North America Lower Canada Battle of
</item 3>

<item 4>
Battle of Lake Erie
USS United States vs HMS Macedonian Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie Battle of Fort Oswego Engagements on Lake Huron
Engagements on Lake Ontario First Battle of Sacket 's Harbor Gulf Coast Action of 13 December 1814 Battle of Lake Borgne Pacific Ocean Action off James Island
Action off Charles Island Nuku Hiva Campaign Battle of Valparaiso ( Capture of USS Essex ) See also : American Indian Wars , Creek War , Napoleonic Wars , and
Tecumseh 's War Portal definition textbooks quotes source texts media news stories Coordinates : 41 ° 39 ′ 44 '' N 82 ° 49 ′ 30 '' W ﻿ / ﻿ 41.66222 ° N 82.82500
° W ﻿ / 41.66222 ; - 82.82500 Retrieved from `` https://en.wikipedia.org/w/index.
</item 4>

<item 5>
Battle of Lake Erie
It was one of the biggest naval battles of the War of 1812 . Contents ( hide ) 1 Background 1.1 1812 1.2 1813 1.3 Blockades of Presque Isle and Amherstburg 2
Battle 3 Casualties 4 Aftermath 5 Reconstructions and memorials 6 Reasons for the American victory 7 Order of battle 8 Footnotes 9 Notes 10 References 11
Further reading 12 External links Background ( edit ) 1812 ( edit ) When the war broke out , the British immediately seized control of Lake Erie . They already
had a small force of warships there : the sloop - of - war Queen Charlotte and the brig General Hunter . The schooner Lady Prevost was under construction and
was put into service a few weeks after the outbreak of war . These vessels were controlled by the Provincial Marine , which was a military transport service and
not a naval service . Nevertheless , the Americans lacked any counter to the British armed vessels .
</item 5>

<item 6>
Battle of Lake Erie
James Massacre Missouri Battle of Credit Island Battle of the Sink Hole British North America Lower Canada Battle of the Chateauguay First Battle of Lacolle
Mills Second Battle of Lacolle Mills Upper Canada Battle of Beaver Dams Battle of Chippawa Battle of Cook 's Mills Battle of Crysler 's Farm Battle of Fort
George Battle of Frenchman 's Creek Battle of Longwoods Battle of Lundy 's Lane Battle of Malcolm 's Mills Battle of Queenston Heights Battle of Stoney Creek
Battle of the Thames Battle of York Capture of Fort Erie Raid on Elizabethtown Raid on Port Dover Raid on Gananoque Siege of Fort Erie Spanish Empire Spanish
Florida Battle of Pensacola Naval battles Atlantic Ocean Capture of HMS Boxer Capture of HMS Cyane Capture of HMS Epervier Capture of HMS Frolic Capture of HMS
Penguin Capture of HMS Dominica Capture of USS Argus Capture of USS Chesapeake Capture of USS President Chesapeake Bay Flotilla USS Constitution vs HMS Java
Sinking of HMS Avon Battle of Fayal Sinking of HMS Peacock Sinking of HMS Reindeer USS Constitution vs HMS Guerriere USS United States vs HMS Macedonian
Caribbean Sea Battle of La Guaira Great Lakes Battle of Lake Erie
</item 6>

<item 7>
Battle of Lake Erie
The Americans lost 27 killed and 96 wounded , of whom 2 later died . Of the vessels involved , the three most battered ( the American brig Lawrence and the
British ships Detroit and Queen Charlotte ) were converted into hospital ships . A gale swept the lake on 13 September and dismasted Detroit and Queen Charlotte
, further shattering the already battered ships . Once the wounded had been ferried to Erie , Lawrence was restored to service for 1814 , but the two British
ships were effectively reduced to hulks . Aftermath ( edit ) This section needs additional citations for verification . Please help improve this article by
adding citations to reliable sources . Unsourced material may be challenged and removed . ( September 2017 ) ( Learn how and when to remove this template
message ) Perry 's vessels and prizes were anchored and hasty repairs were underway near West Sister Island when Perry composed his now famous message to
Harrison . Scrawled in pencil on the back of an old envelope , Perry wrote : Dear General : We have met the enemy and they are ours . Two ships , two brigs ,
one schooner and one sloop .
</item 7>

<item 8>
The text lists naval campaigns and engagements of the War of 1812 across the Atlantic, East Coast, Great Lakes and St. Lawrence, West Indies and Gulf Coast, and
Pacific. It names ship-to-ship battles and captures, including Lake Erie, Sacket’s Harbor, Lake Borgne, and USS Essex’s capture at Valparaiso, and points to
related conflicts such as the Creek War, Tecumseh’s War, and the Napoleonic Wars.
</item 8>

<item 9>
Battle of Lake Erie
Naval campaigns of the War of 1812 Atlantic Ocean USS Essex vs HMS Alert USS Constitution vs HMS Guerriere Capture of HMS Frolic USS United States vs HMS
Macedonian USS Constitution vs HMS Java Sinking of HMS Peacock Rappahannock River Capture of USS Chesapeake Capture of HMS Dominica Capture of USS Argus Capture
of HMS Boxer Capture of USS Frolic Capture of HMS Epervier Sinking of HMS Reindeer Sinking of HMS Avon Fayal Capture of USS President Capture of HMS Cyane and
HMS Levant Capture of HMS Penguin Capture of East India Company ship Nautilus East Coast Chesapeake Bay Alexandria Baltimore Hampden Fort Peter Great Lakes /
Saint Lawrence River Lake Ontario 1st Sacket 's Harbor York Fort George 2nd Sacket 's Harbor Lake Erie Fort Oswego Lake Huron Lake Champlain West Indies / Gulf
Coast La Guaira 1st Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St.
</item 9>

<item 10>
Battle of Lake Erie
Fort Bowyer Action of 13 December 1814 Lake Borgne New Orleans Fort St. Philip 2nd Fort Bowyer Pacific Ocean James Island Charles Island Nuku Hiva Downes
Expedition Porter Expedition Typee Valley Valparaiso ( Capture of USS Essex ) Seringapatam Mutiny Action of 9 May 1814 The Battle of Lake Erie , sometimes
called the Battle of Put - in - Bay , was fought on 10 September 1813 , in Lake Erie off the coast of Ohio during the War of 1812 . Nine vessels of the United
States Navy defeated and captured six vessels of the British Royal Navy . This ensured American control of the lake for the rest of the war , which in turn
allowed the Americans to recover Detroit and win the Battle of the Thames to break the Indian confederation of Tecumseh . It was one of the biggest naval
battles of the War of 1812 .
</item 10>
=== END ITEM id=1bffe1e8651e8150 ===

=== ITEM id=2f6bb433a42e0bba ===
Gold sentences and facts:
Sentence 0: It is also known as an `` immediate hemolytic transfusion reaction '' . This is a medical emergency as it results from rapid destruction of the donor red blood cells by host antibodies ( IgG , IgM ) . It is usually related to ABO blood group incompatibility - the most severe of which often involves group A red cells being given to a patient with group O type blood . Properdin then binds to complement C3 in the donor blood , facilitating the reaction through the alternate pathway cascade . The donor cells also become coated with IgG and are subsequently removed by macrophages in the reticuloendothelial system ( RES ) . Jaundice and disseminated intravascular coagulation ( DIC ) may also occur . The most common cause is clerical error ( i.e. the wrong unit of blood being given to the patient ) .
  Fact 0.0: The condition is also known as an immediate hemolytic transfusion reaction
  Fact 0.1: The reaction is a medical emergency
  Fact 0.2: The reaction results from rapid destruction of the donor red blood cells by host antibodies
  Fact 0.3: The host antibodies involved are IgG and IgM
  Fact 0.4: The reaction is usually related to ABO blood group incompatibility
  Fact 0.5: The most severe ABO incompatibility often involves group A red cells being given to a patient with group O type blood
  Fact 0.6: Properdin binds to complement C3 in the donor blood
  Fact 0.7: Properdin's binding facilitates the reaction through the alternate pathway cascade
  Fact 0.8: The donor cells become coated with IgG
  Fact 0.9: The IgG-coated donor cells are subsequently removed by macrophages in the reticuloendothelial system (RES)
  Fact 0.10: Jaundice may also occur
  Fact 0.11: Disseminated intravascular coagulation (DIC) may also occur
  Fact 0.12: The most common cause of the reaction is clerical error
  Fact 0.13: Clerical error means the wrong unit of blood being given to the patient

Retrieved items:
<item 1>
Acute hemolytic transfusion reaction
the wrong unit of blood being given to the patient ) . Contents ( hide ) 1 Epidemiology 2 Types 2.1 Intravascular haemolysis 2.2 Extravascular haemolysis 3
Signs and symptoms 4 Pathophysiology 4.1 Immune mediated 4.2 Non-immune mediated 5 Diagnosis 5.1 Differential diagnoses 6 Management 7 Prognosis 8 Epidemiology
9 References Epidemiology ( edit ) In the USA in 2011 there were 48 episodes of acute hemolysis due to a mismatch in ABO blood type ( 1 in 495,207 blood
components transfused ) and 168 episodes of hemolysis due to other causes ( 1 in 124,525 blood components transfused ) . In the UK in 2014 there were 18
episodes of hemolysis due to other causes ( 1 in 147,972 blood components transfused ) . Types ( edit ) Acute hemolytic transfusion reactions are divided into
two types .
</item 1>

<item 2>
Acute hemolytic transfusion reaction
ISBN 1 - 4051 - 3649 - 9 . Jump up ^ Molthan L , Matulewicz TJ , Bansal - Carver B , Benz EJ ( 1984 ) . `` An immediate hemolytic transfusion reaction due to
anti-C and a delayed hemolytic transfusion reaction due to anti-Ce + e : hemoglobinemia , hemoglobinuria and transient impaired renal function '' . Vox Sang. 47
( 5 ) : 348 -- 53 . doi : 10.1111 / j. 1423 - 0410.1984. tb04138. x . PMID 6438912 . Jump up ^ `` The 2011 National Blood Collection and Utilization Survey
Report '' ( PDF ) . Department of Health and Human Services . Retrieved 21 January 2016 . Jump up ^ Bolton - Maggs , PHB ; Poles , D ; et al . ( The Serious
Hazards of Transfusion ( SHOT ) Steering Group ) ( 2015 ) . The 2014 Annual SHOT Report ( 2015 ) ( PDF ) .
</item 2>

<item 3>
Acute hemolytic transfusion reaction
Properdin then binds to complement C3 in the donor blood , facilitating the reaction through the alternate pathway cascade . The donor cells also become coated
with IgG and are subsequently removed by macrophages in the reticuloendothelial system ( RES ) . Jaundice and disseminated intravascular coagulation ( DIC ) may
also occur . The most common cause is clerical error ( i.e. the wrong unit of blood being given to the patient ) .
</item 3>

<item 4>
Acute hemolytic transfusion reaction
Acute hemolytic transfusion reaction - wikipedia Acute hemolytic transfusion reaction Jump to : navigation , search This article may be too technical for most
readers to understand . Please help improve it to make it understandable to non-experts , without removing the technical details . ( January 2018 ) ( Learn how
and when to remove this template message ) An acute hemolytic transfusion reaction ( AHTR ) is a type of transfusion reaction that is associated with hemolysis
. It occurs very soon after the transfusion , and within 24 hrs post-transfusion . It can occur quickly upon transfusing a few milliliters , or up to 1 -- 2
hours post-transfusion . It is also known as an `` immediate hemolytic transfusion reaction '' . This is a medical emergency as it results from rapid
destruction of the donor red blood cells by host antibodies ( IgG , IgM ) . It is usually related to ABO blood group incompatibility - the most severe of which
often involves group A red cells being given to a patient with group O type blood .
</item 4>

<item 5>
Acute hemolytic transfusion reaction
, lactated Ringer 's solution or normal saline ) , and osmotic diuresis with mannitol . If DIC is documented and bleeding requires treatment , transfusions of
frozen plasma , pooled cryoprecipitates for fibrinogen , and / or platelet concentrates may be indicated . In case of non-antibody mediated AHR This does not
require rigorous management . Diuresis induced by an infusion of normal saline until the intense red color of hemoglobinuria ceases is usually adequate
treatment . Prognosis ( edit ) The major complication is that the hemoglobin , released by the destruction of red blood cells , may cause acute kidney failure (
also known as the `` oliguric phase '' ) . About 20 deaths occur annually in the US due to AHTR .
</item 5>

<item 6>
Acute hemolytic transfusion reaction
Types ( edit ) Acute hemolytic transfusion reactions are divided into two types . Intravascular haemolysis ( edit ) The most common cause of this reaction is
ABO blood groups incompatibility , secondary to human errors . In this process , the donor 's red blood cells are destroyed by recipient 's preformed antibodies
through the activation of complement system . Thus , the donor 's red blood cells break open and releases haemoglobin into the blood . The haemoglobin is then
excreted into urine , causes haemoglobinuria . Meanwhile , haemoglobin can also pass through the liver system and process into bilirubin , which accumulates
under the skin , causing jaundice . The release of red blood cell debris and the activation of complement system causes uncontrolled clotting ( disseminated
intravascular coagulation ) and circulatory shock . Extravascular haemolysis ( edit ) In this case , the donor 's red blood cells are recognised and bound to
IgG antibodies .
</item 6>

<item 7>
Acute hemolytic transfusion reaction
Macrophages then recognise these IgG antibodies and engulf these red blood cells , removing the red blood cells from the blood circulation into liver and spleen
. These red blood cells are then destroyed by macrophages in these organs . Anti-Rh antibodies most commonly directed this type of reaction . This process is
much more controlled when compared with intravascular haemolysis . Liver has adequate capacity in processing the bilirubin . Therefore , jaundice rarely occurs
. Signs and symptoms ( edit ) Early signs are fever , low blood pressure , anxiety , and red - colored urine . Late signs are generalized bleeding , caused by
disseminated intravascular coagulation , and low blood pressure . Laboratory assessment is based on a positive Direct Antiglobulin Test ( DAT ) , a decrease in
serum haptoglobin , and an increase in blood levels of the enzyme lactate dehydrogenase and indirect bilirubin levels . Pathophysiology ( edit ) Immune mediated
( edit ) This may be caused by preformed IgM anti-A , anti-B or both .
</item 7>

<item 8>
Acute hemolytic transfusion reaction
About 20 deaths occur annually in the US due to AHTR . Epidemiology ( edit ) Acute hemolytic , immune mediated ( fatal ) - 1 per 250,000 - 600,000 Acute
hemolytic , immune mediated ( nonfatal ) - 1 per 6000 - 33,000 Acute hemolytic , nonimmune - Infrequent References ( edit ) Jump up ^ Covin RB , Evans KS ,
Olshock R , Thompson HW ( 2001 ) . `` Acute hemolytic transfusion reaction caused by anti-Coa '' . Immunohematology. 17 ( 2 ) : 45 -- 9 . PMID 15373591 . Jump
up ^ Hoffbrand , A.V. ; P.A.H. Moss ; J.E. Pettit ( 2006 ) . Essential Haematology : 5th Edition . Blackwell Publishing . ISBN 1 - 4051 - 3649 - 9 .
</item 8>

<item 9>
Acute hemolytic transfusion reaction
It may also be caused by other non-ABO IgG antibodies such as Rh , Kell , or Duffy . The former results in a severe intravascular hemolysis and the latter
typically causes extravascular hemolysis . The reactions are mediated by cytokines like TNF , IL - 8 , monocyte chemoattractant protein , IL - 1 etc .
Non-immune mediated ( edit ) These occur when RBCs are damaged before transfusion , resulting in hemoglobinemia and hemoglobinuria . Clinical symptoms are nil
or milder . Diagnosis ( edit ) Differential diagnoses ( edit ) Disseminated intravascular coagulation Hemolytic anemia Pulmonary edema Septic shock Management (
edit ) In case of antibody - mediated acute hemolytic transfusion reactions , Immediately discontinue the transfusion while maintaining venous access for
emergency management . Anticipate low blood pressure , kidney failure , and DIC . Prophylactic measures to reduce the risk of renal failure may include low -
dose dopamine , vigorous hydration with intravenous crystalloid solutions ( e.g.
</item 9>

<item 10>
Acute hemolytic transfusion reaction
( hide ) Complications of surgical and medical care ( T80 -- T88 , 996 -- 999 ) Transfusion reactions Transfusion hemosiderosis TRALI TACO TA - GvHD FNHTR Acute
/ Delayed hemolytic transfusion reaction Vaccine injury Eczema vaccinatum Other Serum sickness Malignant hyperthermia Herxheimer reaction Graft - versus - host
disease Retrieved from `` https://en.wikipedia.org/w/index.php?title=Acute_hemolytic_transfusion_reaction&oldid=821817747 '' Categories : Transfusion reactions
Complications of surgical and medical care Transfusion medicine Hidden categories : Wikipedia articles that are too technical from January 2018 All articles
that are too technical Articles needing expert attention from January 2018 All articles needing expert attention Talk Contents About Wikipedia Add links This
page was last edited on 22 January 2018 , at 20 : 29 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may
apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc.
</item 10>
=== END ITEM id=2f6bb433a42e0bba ===

=== ITEM id=252211717ab6cb54 ===
Gold sentences and facts:
Sentence 0: Originally introduced as a classical tax system , in which companies were subject to tax on their profits and companies ' shareholders were also liable to income tax on the dividends that they received , the first major amendment to corporation tax saw it move to a dividend imputation system in 1973 , under which an individual receiving a dividend became entitled to an income tax credit representing the corporation tax already paid by the company paying the dividend . The classical system was reintroduced in 1999 , with the abolition of advance corporation tax and of repayable dividend tax credits . Another change saw the single main rate of tax split into three . Tax competition between jurisdictions reduced the main corporate tax rate from 28 % in 2008 - 2010 to a flat rate of 20 % as of April 2015 .
  Fact 0.0: Corporation tax was originally introduced as a classical tax system.
  Fact 0.1: Under the classical system, companies were subject to tax on their profits.
  Fact 0.2: Under the classical system, companies' shareholders were also liable to income tax on the dividends they received.
  Fact 0.3: The first major amendment to corporation tax moved it to a dividend imputation system in 1973.
  Fact 0.4: Under the 1973 dividend imputation system, an individual receiving a dividend became entitled to an income tax credit representing the corporation tax already paid by the company paying the dividend.
  Fact 0.5: The classical system was reintroduced in 1999.
  Fact 0.6: The 1999 reintroduction of the classical system came with the abolition of advance corporation tax.
  Fact 0.7: The 1999 reintroduction of the classical system came with the abolition of repayable dividend tax credits.
  Fact 0.8: Another change saw the single main rate of tax split into three.
  Fact 0.9: Tax competition between jurisdictions reduced the main corporate tax rate from 28% in 2008-2010.
  Fact 0.10: Tax competition between jurisdictions reduced the main corporate tax rate to a flat rate of 20% as of April 2015.

Retrieved items:
<item 1>
United Kingdom corporation tax
The 1979 Conservative Budget of Geoffrey Howe cut the small companies ' rate to 40 % , followed by a further cut in the 1982 Budget to 38 % . The Budgets of
1983 -- 1988 saw sharp cuts in both main and small companies ' rates , falling to 35 % and 25 % respectively . Budgets between 1988 and 2001 brought further
falls to a 30 % main rate and 19 % small companies ' rates . From April 1983 to March 1997 the small companies ' rate was pegged to the basic rate of income tax
. During the 1980s there was briefly a higher rate of tax imposed for capital profits . Chancellor Gordon Brown 's 1999 Budget introduced a 10 % starting rate
for profits from £ 0 to £ 10,000 , effective from April 2000 . Marginal relief applied meaning companies with profits of between £ 10,000 and £ 50,000 paid a
rate between the starting rate and the small companies ' rate ( 19 % in 2000 ) .
</item 1>

<item 2>
United Kingdom corporation tax
, HMRC . Retrieved 13 April 2007 Jump up ^ `` Corporation Tax rates and reliefs - GOV.UK '' . www.gov.uk . Retrieved 2015 - 10 - 13 . ^ Jump up to :
Metallgesellschaft Ltd and Others , Hoechst AG , Hoechst UK Ltd and Commissioners of Inland Revenue , H.M. Attorney General ECJ Cases C - 397 / 98 and 410 / 98
( Joined cases ) , European Court of Justice , 2001 . Retrieved 9 May 2007 Jump up ^ `` Corporation Tax Statistics : Analyses of Corporation Tax receipts and
liabilities '' ( PDF ) . HM Revenue & Customs . 28 August 2015 . p. 8 . Archived from the original ( PDF ) on 4 March 2016 . ^ Jump up to : Corporate Tax ( PDF
) Jump up ^ Daunton , Martin ( 2002 ) . Just Taxes : The Politics of Taxation in Britain , 1914 -- 1979 ( pdf ) . Cambridge University Press .
</item 2>

<item 3>
United Kingdom corporation tax
Corporate Tax rates for Financial Years 2008 -- 2014 2008 -- 2010 2011 2012 2013 2014 Small profits rate 21 % 20 % 20 % 20 % 20 % Small profits upper limit £
300,000 £ 300,000 £ 300,000 £ 300,000 £ 300,000 Marginal relief limits £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £
1,500,000 £ 300,001 -- £ 1,500,000 Main rate 28 % 26 % 24 % 23 % 21 % Notes : The bands shown on the right hand side are divided by one plus the number of
associates ( usually the only associates a company has are fellow group companies , but the term is more widely defined ) The reduced rates do not apply to
close investment holding companies ( companies controlled by fewer than 5 people ( plus associates ) or by their directors / managers , whose main activity is
the holding of investments ) .
</item 3>

<item 4>
United Kingdom corporation tax
The 2002 Budget cut the starting rate to zero , with marginal relief applying in the same way . This caused a significant increase in the number of companies
being incorporated , as businesses that had operated as self - employed , paying income tax on profits from just over £ 5,000 , were attracted to the
corporation tax rate of 0 % on income up to £ 10,000 . Previously self - employed individuals could now distribute profits as dividend payments rather than
salaries . For companies with profits under £ 50,000 the corporation tax rate varied between 0 % and 19 % . Because dividend payments come with a basic rate tax
credit , provided the recipient did not earn more than the basic rate allowance , no further tax would be paid . The number of new companies being formed in
2002 -- 2003 reached 325,900 , an increase of 45 % on 2001 -- 2002 . The fact that individuals operating in this manner could potentially pay no tax at all was
felt by the government to be unfair tax avoidance , and the 2004 Budget introduced a Non-Corporate Distribution Rate .
</item 4>

<item 5>
United Kingdom corporation tax
Another change saw the single main rate of tax split into three . Tax competition between jurisdictions reduced the main corporate tax rate from 28 % in 2008 -
2010 to a flat rate of 20 % as of April 2015 . The UK government faced problems with its corporate tax structure , including European Court of Justice
judgements that aspects of it are incompatible with EU treaties . Tax avoidance schemes marketed by the financial sector have also proven an irritant , and been
countered by complicated anti-avoidance legislation . The complexity of the corporation tax system is a recognised issue . The Labour government , supported by
the Opposition parties , carried through wide - scale reform from the Tax Law Rewrite project , resulting in the Corporation Tax Act 2010 . The tax has slowly
been integrating generally accepted accounting practice , with the corporation tax system in various specific areas based directly on the accounting treatment .
Total net corporation tax receipts in 2014 - 15 were £ 43.0 billion , which was an increase of 7 % from £ 40.3 billion in 2013 - 14 .
</item 5>

<item 6>
United Kingdom corporation tax
Financial year FY17 started on 1 April 2017 and ends on 31 March 2018 . Where a company 's accounting period straddles a financial year in which the corporation
tax rate has changed , the company 's profits for that period are split . For example , a company paying small companies ' rate with its accounting period
running from 1 January to 31 December , and making £ 100,000 of profit in 2007 , would be deemed to have made 90 / 365 * £ 100,000 = £ 24,657.53 in FY06 ( there
are 90 days between 1 January and 31 March ) , and 275 / 365 * £ 100,000 = £ 75 , 34.47 in FY07 , and would pay 19 % on the FY06 portion , and 20 % on the FY07
portion . From 1 April 2010 HM Revenue & Customs updated their terminology and the former Small Companies ' Rate is now called Small Profits Rate .
</item 6>

<item 7>
United Kingdom corporation tax
Relief for expenses ( edit ) Most direct expenses are deductible when calculating taxable income and chargeable gains . Notable exceptions include any costs of
entertaining clients . Companies with investment business may deduct certain indirect expenses known as `` expenses of management '' when calculating their
taxable profits . A similar relief is available for expenses of a life assurance company taxed on the I minus E basis which relate to the company 's basic life
assurance and general annuity business . Donations made to charities are also normally deducted in calculating taxable income if made under Gift Aid . Rates and
payment ( edit ) The 2007 Budget announced a main rate cut from 30 % to 28 % , effective from April 2008 . At the same time , the small companies ' rate was
increased from 19 % to 20 % from April 2007 , 21 % in April 2008 , to stop `` individuals artificially incorporating as small companies to avoid paying their
due share of tax , a practice if left unaddressed would cost the rest of the taxpaying population billions of pounds '' . The rate of corporation tax is
determined by the financial year , which runs from 1 April to the following 31 March .
</item 7>

<item 8>
United Kingdom corporation tax
^ Jump up to : `` Finance Act 1965 ( c. 25 ) , from UK Statute Law Database '' . UK Statutory Publications Office , Ministry of Justice . Retrieved 9 May 2007 .
Jump up ^ Tax Law Rewrite Archived 18 April 2006 at the Wayback Machine. , Her Majesty 's Revenue and Customs ( HMRC ) . Retrieved 17 April 2007 Jump up ^ ( 1 )
Income and Corporation Taxes Act 1988 ( c. 1 ) , Office of Public Sector Information , responsible for the operation of Her Majesty 's Stationery Office ( HMSO
) , ISBN 0 - 10 - 540188 - 9 Jump up ^ Most income tax rules have been rewritten in the Income Tax ( Earnings and Pensions ) Act 2003 , Income Tax ( Trading and
Other Income ) Act 2005 , and Income Tax Act 2007 . ^ Jump up to : Rates of Corporation Tax ( PDF ) Archived 9 June 2007 at the Wayback Machine. , HMRC .
</item 8>

<item 9>
United Kingdom corporation tax
This ensured that where a company paid below the small companies ' rate ( 19 % in 2004 ) , dividend payments made to non-corporates ( for example , individuals
, trusts and personal representatives of deceased persons ) would be subject to additional corporation tax , bringing the corporation tax paid up to 19 % . For
example , a company making £ 10,000 profit , and making a £ 6,000 dividend distribution to an individual and £ 4,000 to another company would pay 19 %
corporation tax on the £ 6,000 . Although this measure substantially reduced the number of small businesses incorporating , the Chancellor in the 2006 Budget
said tax avoidance by small businesses through incorporation was still a major issue , and scrapped the starting rate entirely . Historic tax revenues ( edit )
UK Corporation Tax Revenue 1999 to 2016 10,000 20,000 30,000 40,000 50,000 Taxable profits and accounting profits ( edit ) The starting point for computing
taxable profits is profits before tax ( except for a life assurance company ) .
</item 9>

<item 10>
United Kingdom corporation tax
United Kingdom corporation tax - wikipedia United Kingdom corporation tax Jump to : navigation , search Throughout this article , the unqualified term `` pound
'' and the £ symbol refer to the Pound sterling . UK Corporation Tax Receipts from 1999 - 2016 , both in absolute terms and as % of GDP . Taxation in the United
Kingdom UK Government Departments HM Treasury HM Revenue and Customs UK Government VAT Income tax National Insurance Corporation tax Capital gains tax Motoring
taxes Inheritance tax Stamp duty Insurance Premium Tax Air Passenger Duty PAYE Scottish Government Revenue Scotland Scottish income tax Land and Buildings
Transaction Tax Scottish Landfill Tax Air Departure Tax Welsh Government Welsh Revenue Authority Land Transaction Tax Landfill Disposals Tax Local Government
Council Tax Domestic rates in Northern Ireland Business rates in England Business rates in Wales Business rates in Scotland Business rates in Northern Ireland
In the United Kingdom , corporation tax is a corporate tax levied in the United Kingdom on the profits made by UK - resident companies and on the profits of
entities registered overseas with permanent establishments in the UK . Until 1 April 1965 , companies were taxed at the same income tax rates as individual
taxpayers , with an additional profits tax levied on companies .
</item 10>
=== END ITEM id=252211717ab6cb54 ===

=== ITEM id=574b58b5f504c485 ===
Gold sentences and facts:
Sentence 0: `` The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit . In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
  Fact 0.0: The Glory of Love is a song.
  Fact 0.1: The Glory of Love was written by Billy Hill.
  Fact 0.2: The Glory of Love was recorded by Benny Goodman in 1936.
  Fact 0.3: Benny Goodman's version of The Glory of Love was a number one pop hit.
  Fact 0.4: In 1951, the R&B vocal group The Five Keys had their biggest R&B hit with their version of The Glory of Love.
  Fact 0.5: The Five Keys' version hit number one on the R&B chart for four non-consecutive weeks.
  Fact 0.6: The Five Keys' recording reportedly sold a million copies.
  Fact 0.7: Pressed recordings of The Five Keys' version are very rare.
Sentence 1: `` The Glory of Love '' Single by Benny Goodman Released 1936 Format 78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952
  Fact 1.0: The Glory of Love is a single by Benny Goodman.
  Fact 1.1: The Glory of Love was released in 1936.
  Fact 1.2: The Glory of Love has the format 78 rpm vinyl.
  Fact 1.3: The Glory of Love was recorded in 1936.
  Fact 1.4: The Glory of Love has the genre pop.
  Fact 1.5: The songwriter of The Glory of Love is Billy Hill.
  Fact 1.6: The Benny Goodman singles chronology lists The Glory of Love (1936).
  Fact 1.7: The Five Keys singles chronology lists The Glory of Love (1951).
  Fact 1.8: The Five Keys singles chronology lists Yes Sir, That's My Baby (1952).

Retrieved items:
<item 1>
The Glory of Love (song)
Although The Five Keys recording sold a reported million copies pressed recordings are very rare . Contents ( hide ) 1 Charts 1.1 Weekly charts 2 Otis Redding
version 3 Charts 3.1 Weekly charts 4 References Charts ( edit ) Weekly charts ( edit ) Chart ( 1951 ) Peak position US Billboard Hot Rhythm & Blues Songs Otis
Redding version ( edit ) `` The Glory of Love '' Single by Otis Redding from the album The Dock of the Bay B - side `` I 'm Coming Home '' Released 1967 Format
7 '' vinyl Recorded December , 1967 Genre Soul , Pop Length 2 : 38 Label Volt S419 Songwriter ( s ) Billy Hill Producer ( s ) Steve Cropper Otis Redding singles
chronology `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 ) `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 )
In 1967 , Otis Redding recorded a cover version for his 1968 album , The Dock of the Bay .
</item 1>

<item 2>
God Save the Queen
78 ( 1842 ) . Gaetano Donizetti used this anthem in his opera `` Roberto Devereux '' . Joachim Raff used this anthem in his Jubelouverture , Opus 103 ( 1864 )
dedicated to Adolf , Herzog von Nassau , on the 25th anniversary of his reign . Gioachino Rossini used this anthem in the last scene of his `` Il viaggio a
Reims '' , when all the characters , coming from many different European countries , sing a song which recalls their own homeland . Lord Sidney , bass , sings
`` Della real pianta '' on the notes of `` God Save the King '' . Samuel Ramey used to interpolate a spectacular virtuoso cadenza at the end of the song .
Fernando Sor used the anthem in his 12 Studies , Op. 6 : No. 10 in C Major in the section marked ' Maestoso . ' Arthur Sullivan quotes the anthem at the end of
his ballet Victoria and Merrie England .
</item 2>

<item 3>
The Glory of Love (song)
Redding 's cover became a top 20 hit and reached number nineteen on the Billboard R&B Songs chart and number sixty on the Billboard Hot 100 . Charts ( edit )
Weekly charts ( edit ) Chart ( 1967 ) Peak position US Billboard Hot 100 60 US Billboard Hot Rhythm & Blues Songs 19 References ( edit ) Jump up ^ Whitburn ,
Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 205 . Jump up ^ Propes , Steve ( 1973 ) . Those Oldies But Goodies : A Guide to
50 's Record Collecting . The Macmillan Company , New York . p. 42 . Jump up ^ Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record
Research . p. 486 . Jump up ^ `` allmusic ( ( ( Otis Redding > Awards ) ) ) '' . Billboard . Retrieved 2012 - 11 - 22 .
</item 3>

<item 4>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 4>

<item 5>
The Day the Music Died
ISBN 978 - 0 - 89672 - 365 - 8 . Crouse , Richard ( 2012 ) . Who Wrote The Book of Love ? . Random House Digital . ISBN 978 - 0 - 385 - 67442 - 3 . Everitt ,
Rich ( 2004 ) . Falling Stars : Air Crashes That Filled Rock and Roll Heaven . Harbor House . ISBN 978 - 1 - 891799 - 04 - 4 . McDonald , Les ( 2010 ) . The Day
the Music Died . Xlibris Corporation . ISBN 978 - 1 - 4691 - 1356 - 2 . Jennings , Waylon ; Kaye , Lenny ( 1996 ) . Waylon : An Autobiography . Warner Books .
ISBN 978 - 0 - 446 - 51865 - 9 . Schuck , Raymond ( 2012 ) . Do You Believe in Rock and Roll ? Essays on Don Mclean 's American Pie . McFarland .
</item 5>

<item 6>
The Glory of Love (song)
In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four
non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
</item 6>

<item 7>
The Glory of Love (song)
The Glory of Love ( song ) - wikipedia The Glory of Love ( song ) Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2015 ) ( Learn how and
when to remove this template message ) For the song by Peter Cetera , see Glory of Love . `` The Glory of Love '' Single by Benny Goodman Released 1936 Format
78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The
Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952 ``
The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit .
</item 7>

<item 8>
The Glory of Love (song)
Billboard . Retrieved 2012 - 11 - 22 . Jump up ^ `` Billboard R&B Singles Chart '' . Billboard . Retrieved 2016 - 01 - 27 . Preceded by `` Sixty Minute Man ''
by The Dominoes Billboard Best Selling Retail Rhythm & Blues Records number - one single ( The Five Keys version ) September 22 , 1951 Succeeded by `` Sixty
Minute Man '' by The Dominoes Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Glory_of_Love_(song)&oldid=814898703 '' Categories : 1936 songs
1951 singles Benny Goodman songs Otis Redding songs Songs written by Billy Hill ( songwriter ) Billboard Hot R&B / Hip - Hop Songs number - one singles Paul
McCartney songs Hidden categories : Articles needing additional references from December 2015 All articles needing additional references Music infoboxes with
deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia Norsk Edit links This page was last edited on 11 December 2017 , at 15 :
51 . About Wikipedia
</item 8>

<item 9>
God Save the Queen
Some of these are : The French Marquise de Créquy wrote in her Souvenirs that the words `` Grand Dieu , sauvez le Roi ! '' were indeed composed by Jean -
Baptiste Lully in gratitude for the survival by Louis XIV of an anal fistula operation . ( The surgical knife purpose - made for the occasion is on display in
the Musée d'histoire de la médecine . ) Allegedly , Lully had set words by Marie de Brinon to music , and Créquy claimed that the tune was later plagiarised by
Handel , and so adopted by the British as `` un de leurs air nationaux '' . ( The entire story might have been intended as a joke . ) Translated into Latin
under the title Domine , Salvum Fac Regem , it became the French anthem until 1792 . After the Battle of Culloden , the Hanover dynasty supposedly adopted this
melody as the British anthem . James Oswald was a possible author of the Thesaurus Musicus , so may have played a part in the history of the song , but is not a
strong enough candidate to be cited as the composer of the tune .
</item 9>

<item 10>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 10>
=== END ITEM id=574b58b5f504c485 ===

=== ITEM id=68ea2d34f84bc6d9 ===
Gold sentences and facts:
Sentence 0: Shortwave radio is radio transmission using shortwave radio frequencies . There is no official definition of the band , but the range always includes all of the high frequency band ( HF ) , and generally extends from 1.7 -- 30 MHz ( 176.3 -- 10.0 m ) ; from the high end of the medium frequency band ( MF ) just above the mediumwave AM broadcast band , to the end of the HF band .
  Fact 0.0: Shortwave radio is radio transmission using shortwave radio frequencies.
  Fact 0.1: There is no official definition of the shortwave band.
  Fact 0.2: The shortwave band range always includes the high frequency band (HF).
  Fact 0.3: The shortwave band generally extends from 1.7 to 30 MHz (176.3 to 10.0 m).
  Fact 0.4: The shortwave band extends from the high end of the medium frequency band (MF), just above the mediumwave AM broadcast band.
  Fact 0.5: The shortwave band extends to the end of the HF band.
Sentence 1: Very little infrastructure is required for long - distance two - way communications using shortwave radio . All one needs is a pair of transceivers , each with an antenna , and a source of energy ( such as a battery , a portable generator , or the electrical grid ) . This makes shortwave radio one of the most robust means of communications , which can be disrupted only by interference or bad ionospheric conditions . Modern digital transmission modes such as MFSK and Olivia are even more robust , allowing successful reception of signals well below the noise floor of a conventional receiver .
  Fact 1.0: Very little infrastructure is required for long-distance two-way communication using shortwave radio.
  Fact 1.1: All one needs for shortwave radio communication is a pair of transceivers, each with an antenna.
  Fact 1.2: All one needs for shortwave radio communication is a source of energy, such as a battery, a portable generator, or the electrical grid.
  Fact 1.3: Shortwave radio is one of the most robust means of communications.
  Fact 1.4: Shortwave radio communications can be disrupted only by interference or bad ionospheric conditions.
  Fact 1.5: Modern digital transmission modes such as MFSK and Olivia are even more robust.
  Fact 1.6: Modern digital transmission modes such as MFSK and Olivia allow successful reception of signals well below the noise floor of a conventional receiver.

Retrieved items:
<item 1>
Shortwave radio
See also ( edit ) ALLISS -- a very large rotatable antenna system used in international broadcasting Amateur radio -- also uses the shortwave bands , but with
power levels under 2 kW International broadcasting List of American shortwave broadcasters List of shortwave radio broadcasters Long wave Medium wave Shortwave
bands -- shortwave spectrum allocation shortwave relay station -- the fundamental way in which programmes are broadcast on shortwave SSB -- a method of radio
signal modulation References ( edit ) Jump up ^ Grundig Satellit 400 international / professional ShortWaveRadio.ch Jump up ^ Inconsistent article :
itwissen.info : KW ( Kurzwelle ) SW ( short wave ) Quote : `` ... Der KW - Frequenzbereich ( SW ) liegt zwischen 3 MHz und 30 MHz ( < - 1 . definition ) ...
Kurzwelle ist auch eine sendetechnische Bezeichnung für Rundfunk im Frequenzbereich zwischen 2 , 3 MHz und 26 , 1 MHz ( < - 2 .
</item 1>

<item 2>
Shortwave radio
Continuous wave ( CW ) is on - and - off keying of a carrier , used for Morse code communications and Hellschreiber facsimile - based teleprinter transmissions
. Narrow - band frequency modulation ( NBFM ) is used typically above 20 MHz . Because of the larger bandwidth required , NBFM is commonly used for VHF
communication . Regulations limit the bandwidth of a signal transmitted in the HF bands , and the advantages of frequency modulation are greatest if the FM
signal has a wide bandwidth . NBFM is limited to short - range transmissions due to the multiphasic distortions created by the ionosphere . Digital Radio
Mondiale ( DRM ) is a digital modulation for use on bands below 30 MHz . Radioteletype , fax , digital , slow - scan television and other systems use forms of
frequency - shift keying or audio subcarriers on a shortwave carrier . These generally require special equipment to decode , such as software on a computer
equipped with a sound card .
</item 2>

<item 3>
Shortwave radio
Users ( edit ) Portable shortwave receiver 's digital display tuned to the 75 meter band Some established users of the shortwave radio bands may include :
International broadcasting primarily by government - sponsored propaganda , international news ( for example , the BBC World Service ) or cultural stations to
foreign audiences : the most common use of all . Domestic broadcasting : to widely dispersed populations with few longwave , mediumwave and FM stations serving
them ; or for specialty political , religious and alternative media networks ; or of individual commercial and non-commercial paid broadcasts . Oceanic air
traffic control uses the HF / shortwave band for long distance communication to aircraft over the oceans and poles , which are far beyond the range of
traditional VHF frequencies . Modern systems also include satellite communications , such as ADS - C / CPDLC `` Utility '' stations transmitting messages not
intended for the general public , such as merchant shipping , marine weather , and ship - to - shore stations ; for aviation weather and air - to - ground
communications ; for military communications ; for long - distance governmental purposes , and for other non-broadcast communications .
</item 3>

<item 4>
Shortwave radio
The last WRC took place in 2007 . At WRC - 97 in 1997 , the following bands were allocated for international broadcasting . AM shortwave broadcasting channels
are allocated with a 5 kHz separation for traditional analog audio broadcasting . Metre Band Frequency Range Remarks 120 m 2.3 -- 2.495 MHz tropical band 90 m
3.2 -- 3.4 MHz tropical band 75 m 3.9 -- 4 MHz shared with the North American amateur radio 80m band 60 m 4.75 -- 5.06 MHz tropical band 49 m 5.9 -- 6.2 MHz 41
m 7.2 -- 7.6 MHz shared with the amateur radio 40m band 31 m 9.4 -- 9.9 MHz currently the most heavily used band 25 m 11.6 -- 12.2 MHz 22 m 13.57 -- 13.87 MHz
19 m 15.1 -- 15.8 MHz 16 m 17.48 -- 17.9 MHz 15 m 18.9 -- 19.02 MHz almost unused ,
</item 4>

<item 5>
Shortwave radio
L band S band Digital systems ADR DAB - S DVB - SH S - DMB SDR Commercial radio providers 1worldspace Sirius XM Holdings SiriusXM Canada Codecs AAC AMR - WB+
HDC HE - AAC MPEG - 1 Audio Layer II Subcarrier signals AMSS DirectBand PAD RDS / RBDS SCA / SCMO DARC Related topics Technical ( audio ) Audio data compression
Audio signal processing Technical ( AM stereo formats ) Belar C - QUAM Harris Kahn - Hazeltine Magnavox Technical ( emission ) AM broadcasting AM expanded band
Cable radio Digital radio Error detection and correction FM broadcast band FM broadcasting Multipath propagation Shortwave relay station Cultural History of
radio International broadcasting Comparison of radio systems Radio spectrum ( ITU ) ELF 3 Hz / 100 Mm 30 Hz / 10 Mm SLF 30 Hz / 10 Mm 300 Hz / 1 Mm ULF 300 Hz /
1 Mm 3 kHz / 100 km VLF 3 kHz / 100 km 30 kHz / 10 km LF 30 kHz / 10 km 300 kHz / 1 km MF 300 kHz / 1 km 3 MHz / 100 m HF
</item 5>

<item 6>
Shortwave radio
9 MHz 15 m 18.9 -- 19.02 MHz almost unused , could become a DRM band 13 m 21.45 -- 21.85 MHz 11 m 25.6 -- 26.1 MHz may be used for local DRM broadcasting
Although countries generally follow the table above , there may be small differences between countries or regions . For example , in the official bandplan of
the Netherlands , the 49 m band starts at 5.95 MHz , the 41 m band ends at 7.45 MHz , the 11 m band starts at 25.67 MHz , and the 120 , 90 and 60 m bands are
absent altogether . Additionally , international broadcasters sometimes operate outside the normal WRC - allocated bands or use off - channel frequencies . This
is done for practical reasons , or to attract attention in crowded bands ( 60m , 49m , 40m , 41m , 31m , 25m ) . The new digital audio broadcasting format for
shortwave DRM operates 10 kHz or 20 kHz channels .
</item 6>

<item 7>
Shortwave radio
`` Short - Wave Radio Telephone is Success in Tests '' Popular Mechanics , July 1931 , mid page experiments carried out for the French and British governments
Que Escuchar en la Onda Corta en Español website Time signal stations Longwave BPC BPL BSF DCF77 JJY RBU RJH66 ( Beta ) RTZ MSF TDF WWVB Shortwave BPM BSF CHU
HD2IOA HLA JN53DV RWM ROA WWV WWVH ( Hawaii , USA ) YVTO VHF / FM / UHF RDS CT Satellite GPS BeiDou Galileo GLONASS IRNSS QZSS Defunct HBG NAA WWVL OMA OLB5 VNG
Y3S Analog and digital audio broadcasting Terrestrial Radio modulation AM FM COFDM Frequency allocations LW ( LF ) MW ( MF ) SW ( HF ) VHF ( low / mid / high )
L band ( UHF ) Digital systems CAM - D DAB / DAB+ DRM / DRM+ FMeXtra HD Radio CDR DVB - T2 Lite Satellite Frequency allocations C band K band L band S band
Digital systems ADR DAB - S DVB - SH S - DMB
</item 7>

<item 8>
Shortwave radio
This is called skywave or `` skip '' propagation . Thus shortwave radio can be used for very long distance communication , in contrast to radio waves of higher
frequency which travel in straight lines ( line - of - sight propagation ) and are limited by the visual horizon , about 40 miles ( 64 km ) . Shortwave radio is
used for broadcasting of voice and music to shortwave listeners over very large areas ; sometimes entire continents or beyond . It is also used for military
over-the - horizon radar , diplomatic communication , and two - way international communication by amateur radio enthusiasts for hobby , educational and
emergency purposes , as well as for long distance aviation and marine communications .
</item 8>

<item 9>
Shortwave radio
Shortwave radio received its name because the wavelengths in this band are relatively shorter than 200 m ( 1,500 kHz ) which marked the original upper limit of
the medium frequency band first used for radio communications . The broadcast medium wave band now extends above the 200 m / 1,500 kHz limit , and the amateur
radio 1.8 MHz -- 2.0 MHz band ( known as the `` top band '' ) is the lowest - frequency band considered to be ' shortwave ' . Early long distance radio
telegraphy used long waves , below 300 kilohertz ( kHz ) . The drawbacks to this system included a very limited spectrum available for long distance
communication , and the very expensive transmitters , receivers and gigantic antennas that were required . It was also difficult to beam the radio wave
directionally with long wave , resulting in a major loss of power over long distances . Prior to the 1920s , the shortwave frequencies above 1.5 MHz were
regarded as useless for long distance communication and were designated in many countries for amateur use .
</item 9>

<item 10>
Shortwave radio
The drawback is the receiver is more complicated , since it must re-create the carrier to recover the signal . Small errors in the detection process greatly
affect the pitch of the received signal . As a result , single sideband is not used for music or general broadcast . Single sideband is used for long - range
voice communications by ships and aircraft , Citizen 's Band , and amateur radio operators . Lower sideband ( LSB ) is generally used below 9 MHz and USB (
upper sideband ) above 9 MHz . Vestigal sideband transmits the carrier and one complete sideband , but filters out the other sideband . It is a compromise
between AM and SSB , enabing simple receivers to be used , but requires almost as much transmitter power as AM . One advantage is only half the bandwidth of an
AM signal is used . It can be heard in the transmission of certain radio time signal stations . Vestigial sideband is used for over the air Television
Broadcasts both analog and digital .
</item 10>
=== END ITEM id=68ea2d34f84bc6d9 ===
