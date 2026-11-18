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

=== ITEM id=ecc679fce902b489 ===
Gold sentences and facts:
Sentence 0: German autobahns have no federally mandated speed limit for some classes of vehicles . However , limits are posted ( and enforced ) in areas that are urbanized , substandard , accident - prone , or under construction . On speed - unrestricted stretches , an advisory speed limit ( Richtgeschwindigkeit ) of 130 kilometres per hour ( 81 mph ) applies . While going faster is not illegal as such in the absence of a speed limit , it can cause an increased liability in the case of an accident ; courts have ruled that an `` ideal driver '' who is exempt from absolute liability for `` inevitable '' tort under the law would not exceed Richtgeschwindigkeit .
  Fact 0.0: German autobahns have no federally mandated speed limit for some classes of vehicles.
  Fact 0.1: Speed limits are posted in areas that are urbanized.
  Fact 0.2: Speed limits are posted in areas that are substandard.
  Fact 0.3: Speed limits are posted in areas that are accident-prone.
  Fact 0.4: Speed limits are posted in areas that are under construction.
  Fact 0.5: Posted speed limits are enforced.
  Fact 0.6: On speed-unrestricted stretches, an advisory speed limit applies.
  Fact 0.7: The advisory speed limit is called Richtgeschwindigkeit.
  Fact 0.8: The advisory speed limit is 130 kilometres per hour (81 mph).
  Fact 0.9: Going faster is not illegal as such in the absence of a speed limit.
  Fact 0.10: Going faster can cause an increased liability in the case of an accident.
  Fact 0.11: Courts have ruled that an 'ideal driver' is exempt from absolute liability for 'inevitable' tort under the law.
  Fact 0.12: Courts have ruled that an 'ideal driver' would not exceed Richtgeschwindigkeit.

Retrieved items:
<item 1>
Speed limit
This speed zone is scheduled to be activated November 2016 . Roads without speed limits ( edit ) German border crossing sign showing 50 km / h ( 31 mph ) limit
in built - up areas , 100 km / h ( 62 mph ) in rural areas , but a legal advisory 130 km / h ( 81 mph ) limit for the Autobahn ( motorway ) , increasing
liability in the case of an accident from driving faster Just over half of the German autobahns have only an advisory speed limit ( called in German
Richtgeschwindigkeit ) , 15 % have temporary speed limits due to weather or traffic conditions and 33 % have permanent speed limits , according to 2008
estimates . German federal highways and any road outside of towns which is either a dual carriageway or features at least two lanes per direction do have a
general speed limit of 100 km / h . Usually it is reduced to 80 km / h at Allée - streets ( federal highways bordered by trees or bushes on one or both sites )
.
</item 1>

<item 2>
Autobahn
Certain stretches have lower speed limits during wet weather . Some areas have a speed limit of 120 km / h ( 75 mph ) in order to reduce noise pollution during
overnight hours ( usually 10pm -- 6am ) or because of increased traffic during daytime ( 6am -- 8pm ) . Dynamic traffic signs on an Autobahn Some limits were
imposed to reduce pollution and noise . Limits can also be temporarily put into place through dynamic traffic guidance systems that display the according
message . More than half of the total length of the German autobahn network has no speed limit , about one third has a permanent limit , and the remaining parts
have a temporary or conditional limit . Some cars with very powerful engines can reach speeds of well over 300 km / h ( 190 mph ) . Major German car
manufacturers , except Porsche , follow a gentlemen 's agreement by electronically limiting the top speeds of their cars -- with the exception of some top of
the range models or engines -- to 250 km / h ( 155 mph ) .
</item 2>

<item 3>
Autobahn
Truck stop Scandinavian Park off the A 7 Rest area Dammer Berge on the A 1 Rest areas and truck stops are marked several times , starting several kilometres in
advance , and with larger signs that often include icons announcing what kinds of facilities travellers can expect . Speed limits ( edit ) Autobahn with three
separate lanes in each direction and an emergency lane On the autobahns there is an advisory speed limit ( Richtgeschwindigkeit ) of 130 km / h ( unless
otherwise regulated by signs ) . The `` limits no longer apply '' ( Ende aller Streckenverbote ) sign , indicating a return to the default speed , while lifting
all other limits as well ( all limits are indicated by round signs with red border ) . GPS tracks colored according to speed show considerable speed differences
at an autobahn crossing Germany 's autobahns are famous for being among the few public roads in the world without blanket speed limits for cars and motorcycles
. As such , they are important German cultural identifiers , `` ... often mentioned in hushed , reverential tones by motoring enthusiasts and looked at with a
mix of awe and terror by outsiders . '' Some speed limits are implemented on different autobahns .
</item 3>

<item 4>
Autobahn
Some speed limits are implemented on different autobahns . Certain limits are imposed on some classes of vehicles : 60 km / h ( 37 mph ) Buses carrying standing
passengers Motorcycles pulling trailers 80 km / h ( 50 mph ) Vehicles with maximum allowed weight exceeding 3.5 t ( except passenger cars ) Passenger cars and
trucks with trailers Buses 100 km / h ( 62 mph ) Passenger cars pulling trailers certified for 100 km / h Buses certified for 100 km / h not towing trailers
Additionally , speed limits are posted at most on - and off - ramps and interchanges and other danger points like sections under construction or in need of
repair . Where no general limit is required , the advisory speed limit is 130 km / h ( 81 mph ) , referred to in German as the Richtgeschwindigkeit . The
advisory speed is not enforceable ; however , being involved in an accident driving at higher speeds can lead to the driver being deemed at least partially
responsible due to `` increased operating danger '' ( Erhöhte Betriebsgefahr ) .
</item 4>

<item 5>
Speed limit
Feb 2008 . Retrieved 2010 - 10 - 16 . Currently , 52 % of the German motorways do not have a speed limit , 15 % have temporary speed limits due to weather or
traffic conditions and 33 % have permanent speed limits . On unlimited sections there is a 130 km / h recommendation . Jump up ^ `` § 3 StVO : Speed '' .
Bundesministerium der Justiz und für Verbraucherschutz . Retrieved 2014 - 04 - 29 . Diese Geschwindigkeitsbeschränkung gilt nicht ( ... ) auf anderen Straßen
mit Fahrbahnen für eine Richtung , die durch Mittelstreifen oder sonstige bauliche Einrichtungen getrennt sind . Sie gilt ferner nicht auf Straßen , die
mindestens zwei durch Fahrstreifenbegrenzung ( Zeichen 295 ) oder durch Leitlinien ( Zeichen 340 ) markierte Fahrstreifen für jede Richtung haben .
</item 5>

<item 6>
Autobahn
These limiters can be deactivated , so speeds up to 300 km / h ( 190 mph ) might arise on the German autobahn , but due to other traffic , such speeds are
generally not attainable except during certain times like between 10 p.m. and 6 a.m. or on Sundays ( when trucks drivers have to rest by law ) . Furthermore ,
there are certain autobahn sections which are known for having light traffic , making such speeds attainable during most days ( especially some of those located
in Eastern Germany ) . Most unlimited sections of the autobahn are located outside densely populated areas . Vehicles with a top speed less than 60 km / h ( 37
mph ) ( such as quads , low - end microcars , and agricultural / construction equipment ) are not allowed to use the autobahn , nor are motorcycles and scooters
with low engine capacity regardless of top speed ( mainly applicable to mopeds which are typically limited to 25 or 45 km / h anyway ) . To comply with this
limit , heavy - duty trucks in Germany ( e.g. mobile cranes , tank transporters etc . )
</item 6>

<item 7>
Autobahn
The official German term is Bundesautobahn ( plural Bundesautobahnen , abbreviated BAB ) , which translates as `` federal motorway '' . The literal meaning of
the word Bundesautobahn is `` Federal Auto ( mobile ) Track '' . German autobahns have no federally mandated speed limit for some classes of vehicles . However
, limits are posted ( and enforced ) in areas that are urbanized , substandard , accident - prone , or under construction . On speed - unrestricted stretches ,
an advisory speed limit ( Richtgeschwindigkeit ) of 130 kilometres per hour ( 81 mph ) applies . While going faster is not illegal as such in the absence of a
speed limit , it can cause an increased liability in the case of an accident ; courts have ruled that an `` ideal driver '' who is exempt from absolute
liability for `` inevitable '' tort under the law would not exceed Richtgeschwindigkeit . A 2008 estimate reported that 52 % of the autobahn network had only
the advisory speed limit , 15 % had temporary speed limits due to weather or traffic conditions , and 33 % had permanent speed limits .
</item 7>

<item 8>
Autobahn
mobile cranes , tank transporters etc . ) often have a maximum design speed of 62 km / h ( 39 mph ) ( usually denoted by a round black - on - white sign with ``
62 '' on it ) , along with flashing orange beacons to warn approaching cars that they are travelling slowly . There is no general minimum speed but drivers are
not allowed to drive at an unnecessarily low speed as this would lead to significant traffic disturbance and an increased collision risk . Public debate ( edit
) German national speed limits have a historical association with war - time restrictions and deprivations , the Nazi era , and the Soviet era in East Germany .
`` Free driving for free citizens '' ( `` freie Fahrt für freie Bürger '' ) , a slogan promoted by the German Auto Club since the 1970s , is a popular slogan
among those opposing autobahn speed restrictions .
</item 8>

<item 9>
Autobahn
The Federal Road Research Institute ( Bundesanstalt für Straßenwesen ) solicited information about speed regulations on autobahns from the sixteen States and
reported the following , comparing the years 2006 and 2008 : Parameter 2006 2008 Change Autobahn Carriageway - Km 24,735 km 25,240 km + 505 km Advisory Limit
Only 69.2 % 65.5 % - 580 km Variable Limit ( with Advisory Maximum ) 4.2 % 4.1 % - 5 km Permanent or Conditional Speed Limit 26.7 % 30.4 % + 1,090 km Except at
construction sites , the general speed limits , where they apply , are usually between 100 km / h ( 62 mph ) and 130 km / h ( 81 mph ) ; construction sites
usually have a speed limit of 80 km / h ( 50 mph ) but the limit may be as low as 60 km / h ( 37 mph ) . In rare cases , sections may have limits of 40 km / h (
25 mph ) , or on one ramp 30 km / h ( 19 mph ) .
</item 9>

<item 10>
Autobahn
In 2006 , speeds were recorded using automated detection loops in the State of Brandenburg at two points : on a six - lane section of A9 near Niemegk with a 130
km / h ( 81 mph ) advisory speed limit ; and on a four - lane section of A10 bypassing Berlin near Groß Kreutz with a 120 km / h ( 75 mph ) mandatory limit .
</item 10>
=== END ITEM id=ecc679fce902b489 ===

=== ITEM id=a3f942f083d29461 ===
Gold sentences and facts:
Sentence 0: It houses the temporal lobes of the brain and the pituitary gland . A middle fossa craniotomy is one means to surgically remove acoustic neuromas ( vestibular schwannoma ) growing within the internal auditory canal of the temporal bone .
  Fact 0.0: The structure houses the temporal lobes of the brain and the pituitary gland.
  Fact 0.1: A middle fossa craniotomy is one means to surgically remove acoustic neuromas.
  Fact 0.2: Acoustic neuromas are also called vestibular schwannomas.
  Fact 0.3: Acoustic neuromas grow within the internal auditory canal of the temporal bone.

Retrieved items:
<item 1>
Brain
The superior colliculus is part of the midbrain . The pallium is a layer of gray matter that lies on the surface of the forebrain and is the most complex and
most recent evolutionary development of the brain as an organ . In reptiles and mammals , it is called the cerebral cortex . Multiple functions involve the
pallium , including smell and spatial memory . In mammals , where it becomes so large as to dominate the brain , it takes over functions from many other brain
areas . In many mammals , the cerebral cortex consists of folded bulges called gyri that create deep furrows or fissures called sulci . The folds increase the
surface area of the cortex and therefore increase the amount of gray matter and the amount of information that can be stored and processed . The hippocampus ,
strictly speaking , is found only in mammals . However , the area it derives from , the medial pallium , has counterparts in all vertebrates . There is evidence
that this part of the brain is involved in complex events such as spatial memory and navigation in fishes , birds , reptiles , and mammals . The basal ganglia
are a group of interconnected structures in the forebrain .
</item 1>

<item 2>
Middle cranial fossa
Behind the tuberculum sellæ is a deep depression , the sella turcica , containing the fossa hypophyseos , which lodges the hypophysis , and presents on its
anterior wall the middle clinoid processes . The sella turcica is bounded posteriorly by a quadrilateral plate of bone , the dorsum sellae , the upper angles of
which are surmounted by the posterior clinoid processes : these afford attachment to the tentorium cerebelli , and below each is a notch for the abducent nerve
. On either side of the sella turcica is the carotid groove , which is broad , shallow , and curved somewhat like the italic letter f . It begins behind at the
foramen lacerum , and ends on the medial side of the anterior clinoid process , where it is sometimes converted into a foramen ( carotico - clinoid ) by the
union of the anterior with the middle clinoid process ; posteriorly , it is bounded laterally by the lingula . This groove lodges the cavernous sinus and the
internal carotid artery , the latter being surrounded by a plexus of sympathetic nerves .
</item 2>

<item 3>
Middle cranial fossa
On the anterior surface of the petrous portion of the temporal bone are seen the eminence caused by the projection of the superior semicircular canal ; in front
of and a little lateral to this a depression corresponding to the roof of the tympanic cavity ; the groove leading to the hiatus of the facial canal , for the
transmission of the greater superficial petrosal nerve and the petrosal branch of the middle meningeal artery ; beneath it , the smaller groove , for the
passage of the lesser superficial petrosal nerve ; and , near the apex of the bone , the depression for the semilunar ganglion and the orifice of the carotid
canal . See also ( edit ) Anterior cranial fossa Posterior cranial fossa References ( edit ) This article incorporates text in the public domain from the 20th
edition of Gray 's Anatomy ( 1918 ) Jump up ^ Mancall , Elliott L. ; Brock , David G. , eds. ( 2011 ) . `` Cranial Fossae '' . Gray 's Clinical Anatomy .
Elsevier Health Sciences . p. 154 . ISBN 9781437735802 .
</item 3>

<item 4>
Middle cranial fossa
wikipedia.org/w/index.php?title=Middle_cranial_fossa&oldid=808415657 '' Categories : Wikipedia articles incorporating text from the 20th edition of Gray 's
Anatomy ( 1918 ) Skull Hidden categories : Medicine infobox template using GraySubject or GrayPage Talk Contents About Wikipedia Deutsch فارسی Polski Edit links
This page was last edited on 2 November 2017 , at 18 : 54 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may
apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a
non-profit organization . About Wikipedia
</item 4>

<item 5>
Middle cranial fossa
Middle cranial fossa - wikipedia Middle cranial fossa Jump to : navigation , search Middle cranial fossa Base of the skull . Upper surface . ( Middle cranial
fossa is the centermost of the three indentations , in pink and yellow . ) Details Identifiers Latin fossa cranii media TA A02. 1.00. 049 FMA 54369 Anatomical
terminology ( edit on Wikidata ) The middle cranial fossa , deeper than the anterior cranial fossa , is narrow medially and widens laterally to the sides of the
skull . It is separated from the posterior fossa by the clivus and the petrous crest . It is bounded in front by the posterior margins of the lesser wings of
the sphenoid bone , the anterior clinoid processes , and the ridge forming the anterior margin of the chiasmatic groove ; behind , by the superior angles of the
petrous portions of the temporal bones and the dorsum sellæ ; laterally by the temporal squamæ , sphenoidal angles of the parietals , and greater wings of the
sphenoid .
</item 5>

<item 6>
Middle cranial fossa
Lateral parts ( edit ) The lateral parts of the middle fossa are of considerable depth , and support the temporal lobes of the brain . They are marked by
depressions for the brain convolutions and traversed by furrows for the anterior and posterior branches of the middle meningeal vessels . These furrows begin
near the foramen spinosum , and the anterior runs forward and upward to the sphenoidal angle of the parietal , where it is sometimes converted into a bony canal
; the posterior runs lateralward and backward across the temporal squama and passes on to the parietal near the middle of its lower border . The following
apertures are also to be seen . In front is the superior orbital fissure , bounded above by the small wing , below , by the great wing , and medially , by the
body of the sphenoid ; it is usually completed laterally by the orbital plate of the frontal bone .
</item 6>

<item 7>
Middle cranial fossa
It is traversed by the squamosal , sphenoparietal , sphenosquamosal , and sphenopetrosal sutures . It houses the temporal lobes of the brain and the pituitary
gland . A middle fossa craniotomy is one means to surgically remove acoustic neuromas ( vestibular schwannoma ) growing within the internal auditory canal of
the temporal bone . Middle cranial fossa surgical anatomy as demonstrated in a right cadaver temporal bone by Dr Jack M Kartush - view from above . Contents (
hide ) 1 Middle part 2 Lateral parts 3 See also 4 References 5 Additional images 6 External links Middle part ( edit ) The middle part of the fossa presents ,
in front , the chiasmatic groove and tuberculum sellae ; the chiasmatic groove ends on either side at the optic foramen , which transmits the optic nerve and
ophthalmic artery to the orbital cavity . Behind the optic foramen the anterior clinoid process is directed backward and medialward and gives attachment to the
tentorium cerebelli .
</item 7>

<item 8>
Middle cranial fossa
p. 154 . ISBN 9781437735802 . Additional images ( edit ) Middle cranial fossa at human foetus Base of skull Middle cranial fossa Middle cranial fossa Middle
cranial fossa External links ( edit ) Wikimedia Commons has media related to Middle cranial fossa . Anatomy photo : 22 : os - 0802 at the SUNY Downstate Medical
Center Compound structures of skull Neurocranium Calvaria Diploë Asterion Pterion Stephanion Inion Bregma Lambda Fossae anterior middle posterior cranial cavity
Base of skull Fontanelle anterior posterior sphenoidal mastoid Facial skeleton Nasion Gonion Both dacryon zygomatic arch temporal fossa infratemporal fossa
pterygomaxillary fissure pterygopalatine fossa Foramina of the skull ( and canals , fissures , meatus , and hiatus ) Anterior cranial fossa to Orbit : ethmoidal
foramina anterior posterior to Nasal cavity : olfactory foramina ( CN - I ) foramen cecum Middle cranial fossa to Orbit : optic canal ( CN - II ) superior
orbital fissure ( CN - III ,
</item 8>

<item 9>
Middle cranial fossa
It transmits to the orbital cavity the oculomotor , the trochlear , the ophthalmic division of the trigeminal , and the abducent nerves , some filaments from
the cavernous plexus of the sympathetic , and the orbital branch of the middle meningeal artery ; and from the orbital cavity a recurrent branch from the
lacrimal artery to the dura mater , and the ophthalmic veins . Behind the medial end of the superior orbital fissure is the foramen rotundum , for the passage
of the maxillary nerve . Behind and lateral to the foramen rotundum is the foramen ovale , which transmits the mandibular nerve , the accessory meningeal artery
, and the lesser superficial petrosal nerve . Medial to the foramen ovale is the foramen Vesalii , which varies in size in different individuals , and is often
absent ; when present , it opens below at the lateral side of the scaphoid fossa , and transmits a small vein .
</item 9>

<item 10>
Middle cranial fossa
Lateral to the foramen ovale is the foramen spinosum , for the passage of the middle meningeal vessels , and a recurrent branch from the mandibular nerve .
Medial to the foramen ovale is the foramen lacerum ; in the fresh state the lower part of this aperture is filled up by a layer of fibrocartilage , while its
upper and inner parts transmit the internal carotid artery surrounded by a plexus of sympathetic nerves . The nerve of the pterygoid canal and a meningeal
branch from the ascending pharyngeal artery pierce the layer of fibrocartilage .
</item 10>
=== END ITEM id=a3f942f083d29461 ===

=== ITEM id=da77a90a1ec04afe ===
Gold sentences and facts:
Sentence 0: Cast Character Rank / Position Seasons Notes Regular Recurring Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 , 17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 -- 5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D. Wong George Huang FBI Special Agent 4 -- 12 2 -- 3 13 -- 15 , 17 Diane Neal Casey Novak Senior Assistant District Attorney 5 -- 9 13 12 Adam Beach Chester Lake Detective 9 8 Michaela McManus Kim Greylek Assistant District Attorney 10 Danny Pino Nick Amaro Detective 13 -- 16 Kelli Giddish Amanda Rollins Detective 13 -- Raúl Esparza Rafael Barba Assistant District Attorney 15 -- 14 Peter Scanavino Dominick Carisi Jr . Detective 16 -- 16
  Fact 0.0: Christopher Meloni plays Elliot Stabler.
  Fact 0.1: Elliot Stabler is a Detective.
  Fact 0.2: Christopher Meloni's season entry is 1 to 12.
  Fact 0.3: Mariska Hargitay plays Olivia Benson.
  Fact 0.4: Olivia Benson is a Lieutenant.
  Fact 0.5: Mariska Hargitay's season entry starts at season 1.
  Fact 0.6: Richard Belzer plays John Munch.
  Fact 0.7: John Munch is a DA Investigator.
  Fact 0.8: Richard Belzer's season entries are 1 to 15 and 15 and 17.
  Fact 0.9: Dann Florek plays Donald Cragen.
  Fact 0.10: Donald Cragen is a Captain.
  Fact 0.11: Dann Florek's season entries are 1 to 15 and 16.
  Fact 0.12: Michelle Hurd plays Monique Jeffries.
  Fact 0.13: Monique Jeffries is a Detective.
  Fact 0.14: Michelle Hurd's season entry is 1 to 2.
  Fact 0.15: Stephanie March plays Alexandra Cabot.
  Fact 0.16: Alexandra Cabot is an Assistant District Attorney.
  Fact 0.17: Stephanie March's season entries are 2 to 5, 11, 2, 10, 13, and 6.
  Fact 0.18: Ice - T plays Odafin "Fin" Tutuola.
  Fact 0.19: Odafin "Fin" Tutuola is a Detective.
  Fact 0.20: Ice - T's season entry starts at season 2.
  Fact 0.21: Tamara Tunie plays Melinda Warner.
  Fact 0.22: Melinda Warner is a Medical Examiner.
  Fact 0.23: Tamara Tunie's season entries are 7 to 12, 2 to 6, 13 to 17, and 19.
  Fact 0.24: B.D. Wong plays George Huang.
  Fact 0.25: George Huang is an FBI Special Agent.
  Fact 0.26: B.D. Wong's season entries are 4 to 12, 2 to 3, 13 to 15, and 17.
  Fact 0.27: Diane Neal plays Casey Novak.
  Fact 0.28: Casey Novak is a Senior Assistant District Attorney.
  Fact 0.29: Diane Neal's season entries are 5 to 9, 13, and 12.
  Fact 0.30: Adam Beach plays Chester Lake.
  Fact 0.31: Chester Lake is a Detective.
  Fact 0.32: Adam Beach's season entries are 9 and 8.
  Fact 0.33: Michaela McManus plays Kim Greylek.
  Fact 0.34: Kim Greylek is an Assistant District Attorney.
  Fact 0.35: Michaela McManus's season entry is 10.
  Fact 0.36: Danny Pino plays Nick Amaro.
  Fact 0.37: Nick Amaro is a Detective.
  Fact 0.38: Danny Pino's season entry is 13 to 16.
  Fact 0.39: Kelli Giddish plays Amanda Rollins.
  Fact 0.40: Amanda Rollins is a Detective.
  Fact 0.41: Kelli Giddish's season entry starts at season 13.
  Fact 0.42: Raúl Esparza plays Rafael Barba.
  Fact 0.43: Rafael Barba is an Assistant District Attorney.
  Fact 0.44: Raúl Esparza's season entry is 15 to 14.
  Fact 0.45: Peter Scanavino plays Dominick Carisi Jr.
  Fact 0.46: Dominick Carisi Jr. is a Detective.
  Fact 0.47: Peter Scanavino's season entry is 16 to 16.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
It is the current longest running scripted non-animated U.S. primetime TV series since the cancellation of the original Law & Order on May 24 , 2010 and is the
fourth - longest running scripted U.S. primetime TV series on a major broadcast network . Law & Order : Special Victims Unit originally centered almost
exclusively on the detectives of the Special Victims Unit in a fictional version of the 16th Precinct of the New York City Police Department . As the series
progressed , additional supporting characters were added as allies of the detectives in the New York County Manhattan District Attorney 's office ( known as
advisers from the Sex Crimes Bureau ) and the Medical Examiner 's office . Certain episodes will go into detail about detectives ' personal lives and how they
may or may not tie into the crimes dealt with during the show . Typical episodes follow the detectives and their colleagues as they investigate and prosecute
sexually based offenses , child abuse , and crimes against the elderly . The show starred Christopher Meloni as Detective Elliot Stabler and Mariska Hargitay as
Detective Olivia Benson for its first twelve seasons until the former left the cast , unable to come to an agreement on his contract .
</item 1>

<item 2>
Law & Order: Special Victims Unit
The series follows a unit of investigators in Moscow whose job is to investigate crimes of a sexual nature . The series airs on NTV and is produced by Pavel
Korchagin , Felix Kleiman , and Edward Verzbovski and directed by Dmitry Brusnikin . The screenplays are written by Sergei Kuznvetsov , Elena Karavaeshnikova ,
and Maya Shapovalova . Casting ( edit ) Cast of Law & Order : Special Victims Unit Cast of seasons 4 -- 5 ( 2002 -- 04 ) , from left : Ice - T , Richard Belzer
, Mariska Hargitay , Dann Florek , Christopher Meloni , B.D.
</item 2>

<item 3>
*Law & Order: Special Victims Unit* is a New York City-set television spin-off created by Dick Wolf and produced by Wolf Films. It is filmed in New York,
features multiple main characters, is based on actual events, and has audio description.
</item 3>

<item 4>
Law & Order: Special Victims Unit
Series overview ( edit ) In the criminal justice system , sexually based offenses are considered especially heinous . In New York City , the dedicated
detectives who investigate these vicious felonies are members of an elite squad known as the Special Victims Unit . These are their stories . -- Opening
narration spoken by Steven Zirnkilton Based out of the New York City Police Department 's 16th precinct in Manhattan , Law & Order : Special Victims Unit delves
into the dark side of the New York underworld as the detectives of a new elite force , the Special Victims Unit ( SVU for short ) , investigate and prosecute
various sexually - oriented crimes including rape , pedophilia , and domestic violence . They also investigate the abuses of children , the disabled and elderly
victims of non-sexual crimes who require specialist handling . All the while trying to balance the effects of the investigation on their own lives . Its stories
also touch on the political and societal issues associated with gender identity , sexual preferences , and equality rights . While the victim is often murdered
, this is not always the case , and victims frequently play prominent roles in episodes .
</item 4>

<item 5>
Created by Dick Wolf, *Law & Order: Special Victims Unit* is an NBC police and legal drama that premiered September 20, 1999. Set in Manhattan’s fictional 16th
Precinct, it follows detectives and prosecutors handling sex crimes, child abuse, and other vulnerable-victim cases, often inspired by real events. Mariska
Hargitay won a 2006 Emmy. The series became a ratings hit despite an early Friday slot, later shifting among weeknights. Fil
</item 5>

<item 6>
Law & Order: Special Victims Unit
1999 ( 1999 - 09 - 20 ) -- present Chronology Related shows Law & Order franchise External links Website Law & Order : Special Victims Unit ( often abbreviated
to Law & Order : SVU or just SVU ) is an American police procedural , legal , crime drama television series set in New York City , where it is also primarily
produced . In the style of the original Law & Order , episodes are often `` ripped from the headlines '' or loosely based on real crimes that have received
media attention . Created by Dick Wolf , the series premiered on NBC on September 20 , 1999 , as the second series in Wolf 's successful Law & Order franchise .
Law & Order : Special Victims Unit has been nominated for and won numerous awards , including the 2006 Emmy for Outstanding Lead Actress in a Drama Series for
Mariska Hargitay , the first , and to date only , Emmy to be received by a regular on any Law & Order series . It is the current longest running scripted
non-animated U.S.
</item 6>

<item 7>
Law & Order: Special Victims Unit
External links ( edit ) Wikimedia Commons has media related to Law & Order : Special Victims Unit . Wikiquote has quotations related to : Law & Order : Special
Victims Unit Official website on NBC Law & Order : Special Victims Unit on IMDb Law & Order : Special Victims Unit at TV.com `` List of Law & Order : SVU
Episodes '' . TV Guide . Law & Order : Special Victims Unit on MyNetworkTV Law & Order franchise Series Law & Order ( 1990 -- 2010 ) Law & Order : Special
Victims Unit ( 1999 -- present ) Law & Order : Criminal Intent ( 2001 -- 2011 ) Law & Order : Trial by Jury ( 2005 -- 2006 ) Law & Order : LA ( 2010 -- 2011 )
Law & Order True Crime ( 2017 -- present ) Adaptations Paris enquêtes criminelles ( 2007 -- 2008 ) Law & Order : UK ( 2009 -- 2014 ) Video games Law & Order :
Dead on the Money Law & Order : Double or Nothing Law & Order : Justice Is Served Law & Order : Criminal Intent Law & Order : Legacies Related series Homicide :
</item 7>

<item 8>
Law & Order: Special Victims Unit
Cast and characters ( edit ) Main article : List of Law & Order : Special Victims Unit characters Cast Character Rank / Position Seasons Notes Regular Recurring
Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 ,
17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 --
5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D.
</item 8>

<item 9>
Law & Order: Special Victims Unit
The unit also works with the Manhattan District Attorney 's office as they prosecute cases and seek justice for SVU 's victims and survivors with precision and
a passion to win and bring closure to the intense investigations . The series often uses stories that are `` ripped from the headlines '' or based on real
crimes . Such episodes take a real crime and fictionalize it by changing some details . Originally the show focused around the detective pairings of Elliot
Stabler & Olivia Benson and John Munch & Brian Cassidy . Stabler is a seasoned veteran of the unit who has seen it all and tries his best to protect his family
from the horrors he sees every day . His partner Benson 's difficult past as the child of a rape victim is the reason she joined the unit . Backing them up is
John Munch , and his first partner Brian Cassidy . Munch is a transfer from Baltimore 's homicide unit , who brings his acerbic wit , conspiracy theories , and
street - honed investigative skills . Whilst Cassidy is young and eager to learn from his fellow detectives . These two detective teams received support from
Detectives Monique Jeffries and Ken Briscoe .
</item 9>

<item 10>
*Law & Order: Special Victims Unit* follows elite NYPD detectives based at Manhattan’s 16th Precinct as they investigate sexual offenses—including rape, child
abuse, and domestic violence—and other cases involving vulnerable victims. Working with the Manhattan District Attorney’s office, they pursue justice while
coping with the investigations’ effects on their lives. Stories often draw on real crimes and address social and political issues. Early central partners were
Elliot Stabler and Olivia Benson, joined by John Munch and Brian Cassidy, with
</item 10>
=== END ITEM id=da77a90a1ec04afe ===

=== ITEM id=70a5f8d86cb84564 ===
Gold sentences and facts:
Sentence 0: In the aftermath of the disputed 1824 presidential election , the separate factions took on many characteristics of parties in their own right . Adams ' supporters , in league with Clay , favored modernization , banks , industrial development and federal spending for roads and other internal improvements , which the Old Republicans and the Jackson men usually opposed . Writing in his personal journal on December 13 , 1826 , President Adams noted the difficulty he faced in attempting to be nonpartisan in appointing men to office :
  Fact 0.0: After the disputed 1824 presidential election, the separate factions took on many characteristics of parties in their own right.
  Fact 0.1: Adams' supporters, in league with Clay, favored modernization.
  Fact 0.2: Adams' supporters, in league with Clay, favored banks.
  Fact 0.3: Adams' supporters, in league with Clay, favored industrial development.
  Fact 0.4: Adams' supporters, in league with Clay, favored federal spending for roads and other internal improvements.
  Fact 0.5: The Old Republicans and the Jackson men usually opposed these positions.
  Fact 0.6: President Adams wrote in his personal journal on December 13, 1826.
  Fact 0.7: President Adams noted in his personal journal on December 13, 1826 the difficulty he faced in attempting to be nonpartisan in appointing men to office.

Retrieved items:
<item 1>
Democratic-Republican Party
Thomas Jefferson wrote on the state of party politics in the early 1820s : An opinion prevails that there is no longer any distinction , that the republicans &
Federalists are completely amalgamated but it is not so . The amalgamation is of name only , not of principle . All indeed call themselves by the name of
Republicans , because that of Federalists was extinguished in the battle of New Orleans . But the truth is that finding that monarchy is a desperate wish in
this country , they rally to the point which they think next best , a consolidated government . Their aim is now therefore to break down the rights reserved by
the constitution to the states as a bulwark against that consolidation , the fear of which produced the whole of the opposition to the constitution at its birth
. Hence new Republicans in Congress , preaching the doctrines of the old Federalists , and the new nick - names of Ultras and Radicals . But I trust they will
fail under the new , as the old name , and that the friends of the real constitution and union will prevail against consolidation , as they have done against
monarchism .
</item 1>

<item 2>
Democratic-Republican Party
HuffPost . Retrieved August 30 , 2017 . Jump up ^ `` Democratic - Republican Party '' . Encyclopædia Britannica . July 20 , 1998 . Retrieved August 30 , 2017 .
The Republicans contended that the Federalists harboured aristocratic attitudes and that their policies placed too much power in the central government and
tended to benefit the affluent at the expense of the common man . Jump up ^ `` Democratic - Republican Party '' . The Gilder Lehrman Institute of American
History . Retrieved December 7 , 2014 . Jump up ^ Susan Dunn ( 2004 ) . Jefferson 's Second Revolution : The Election of 1800 and the Triumph of Republicanism .
Houghton Mifflin Harcourt . p. 279 . ISBN 0618131647 . Jump up ^ Paul Kleppner , et . al .
</item 2>

<item 3>
Compromise of 1877
He was more interested in ensuring that the Radical state government in Louisiana was abandoned than in any southern railroad . Vincent DeSantis argues that the
Republican Party abandoned Southern blacks to the rule of the racist Democratic Party in order to gain the support of Democrats for Hayes ' presidency . In any
case , Reconstruction ended . The dominance of the Democratic Party in the South was cemented with the ascent of the `` Redeemer '' governments that displaced
the Republican governments . After 1877 , support for white supremacy generally caused whites to vote for Democrats and the region became known as the `` Solid
South . '' Until the end of the 19th century , black Republicans continued to elect numerous candidates to local office , although Democrats controlled most
state representative and statewide seats , except for a brief period of fusion governments supported by Republicans and Populists . The majority of white voters
supported national Democratic candidates well into the 20th century before shifting to the Republican Party . This later shift to the Republican party followed
the 1964 Civil Rights Act , which was introduced by Democratic President Lyndon B. Johnson and supported by most Republicans and northern Democrats .
</item 3>

<item 4>
The Democratic-Republican Party began in 1791 as an anti-Federalist faction, including politicians who had opposed the Constitution. Calling themselves
Republicans and later known as Jeffersonian Republicans, they distrusted Federalist republicanism. Jefferson organized a national party to challenge Hamilton’s
growing coalition, opposing his financial programs as monarchical and harmful to republicanism. The party split in 1824: Jacksonians became the Democratic Party
in 1828, while National Republicans were succeeded by the Whigs. Histor
</item 4>

<item 5>
Democratic-Republican Party
Jefferson and Madison often used the terms `` republican '' and `` Republican party '' in their letters . The 1804 Convention of Republican members of Congress
that renominated Jefferson described itself as a `` regular republican caucus '' . The name Democratic - Republican was used by contemporaries only occasionally
. The term `` republican '' was in widespread usage from the 1770s to describe the type of government the break - away colonies wanted to form : a republic of
three separate branches of government derived from some principles and structure from ancient republics ; especially the emphasis on civic duty and the
opposition to corruption , elitism , aristocracy and monarchy . The word is used in the United States Constitution . Legacy ( edit ) Further information : First
Party System and Second Party System A split appeared in the then Republican party during the 1824 elections ( at the end of the Monroe administration ) . When
the election was thrown to the House of Representatives , Henry Clay backed John Quincy Adams to deny the presidency to Andrew Jackson , a longtime political
rival . Jackson defeated Adams in 1828 and in the next election the first Democratic national convention took place in Baltimore , Maryland on May 21 -- 23 ,
1832 .
</item 5>

<item 6>
Democratic-Republican Party
I scarcely know myself which is most to be deprecated , a consolidation , or dissolution of the states . The horrors of both are beyond the reach of human
foresight . In the aftermath of the disputed 1824 presidential election , the separate factions took on many characteristics of parties in their own right .
Adams ' supporters , in league with Clay , favored modernization , banks , industrial development and federal spending for roads and other internal improvements
, which the Old Republicans and the Jackson men usually opposed . Writing in his personal journal on December 13 , 1826 , President Adams noted the difficulty
he faced in attempting to be nonpartisan in appointing men to office : And it is upon the occasion of appointments to office that all the wormwood and the gall
of the old party hatred ooze out . Not a vacancy to any office occurs but there is a distinguished federalist started and pushed home as a candidate to fill it
-- always well qualified , sometimes in an eminent degree , and yet so obnoxious to the Republican party that he can not be appointed without exciting a
vehement clamor against him and against the Administration .
</item 6>

<item 7>
Democratic-Republican Party
Democratic - Republican party - wikipedia Democratic - Republican party Jump to : navigation , search Not to be confused with the Democratic or the Republican
Party of the United States . `` Democratic Republican '' redirects here . For other uses , see Democratic Republican ( disambiguation ) . Democratic -
Republican Party Other name Jefferson Republicans Leader Thomas Jefferson James Madison James Monroe John Quincy Adams Founded 1792 ( 1792 ) Dissolved 1825 (
1825 ) Preceded by Anti-Administration party Succeeded by Democratic Party ( Jacksonians ) National Republican Party ( Anti-Jacksonians ) Ideology Agrarianism
Anti-Federalism Jeffersonianism Republicanism Colors Red White Blue Politics of United States Political parties Elections The Democratic - Republican Party was
an American political party formed by Thomas Jefferson and James Madison between 1791 and 1793 to oppose the centralizing policies of the new Federalist Party
run by Alexander Hamilton , who was secretary of the treasury and chief architect of George Washington 's administration . From 1801 to 1825 , the new party
controlled the presidency and Congress as well as most states during the First Party System .
</item 7>

<item 8>
Democratic-Republican Party
It began in 1791 as one faction in Congress and included many politicians who had been opposed to the new constitution . They called themselves `` Republicans
'' after their ideology , republicanism . They distrusted the Federalist commitment to republicanism . The party splintered in 1824 into the Jacksonian movement
( which became the Democratic Party in 1828 ) and the short - lived National Republican Party ( later succeeded by the Whig Party ) . The term `` Democratic -
Republican '' is used especially by modern political scientists for the first `` Republican Party '' ( as opposed to the modern Republican Party founded in 1854
) . It is also known as the Jeffersonian Republicans . Historians typically use the title `` Republican Party '' . An `` Anti-Administration '' faction met
secretly in the national capital ( Philadelphia ) to oppose Hamilton 's financial programs . Jefferson denounced the programs as leading to monarchy and
subversive of republicanism . Jefferson needed to have a nationwide party to challenge the Federalists , which Hamilton was building up with allies in major
cities .
</item 8>

<item 9>
Democratic-Republican Party
The party selected its presidential candidates in a caucus of members of Congress . They included Thomas Jefferson ( nominated 1796 ; elected 1800 -- 1801 ,
1804 ) , James Madison ( 1808 , 1812 ) and James Monroe ( 1816 , 1820 ) . By 1824 , the caucus system had practically collapsed . After 1800 , the party
dominated Congress and most state governments outside New England . By 1824 , the party was also split four ways and lacked a center , as the First Party System
collapsed . The emergence of the Second Party System in the 1830s realigned the old factions . One remnant followed Andrew Jackson and Martin Van Buren into the
new Democratic Party by 1828 . Another remnant led by John Quincy Adams and Henry Clay formed the National Republicans in 1828 , which developed into the Whig
Party by 1835 .
</item 9>

<item 10>
Democratic-Republican Party
Tompkins 1818 / 1819 30 / 42 5 1820 / 1821 39 / 46 9 1822 / 1823 33 / 48 6 See also ( edit ) Anti-Federalism History of the United States Democratic Party
Jacksonian democracy Jeffersonian democracy List of political parties in the United States References ( edit ) Jump up ^ Ohio History Connection . `` Democratic
- Republican Party '' . Ohio History Central . Retrieved August 30 , 2017 . Democratic - Republicans favored keeping the U.S. economy based on agriculture and
said that the U.S. should serve as the agricultural provider for the rest of the world ( ... ) . Economically , the Democratic - Republicans wanted to remain a
predominantly agricultural nation , ( ... ) . Jump up ^ `` Anti-Federalist vs. Federalist '' . Diffen . Jump up ^ Rich Rubino ( June 13 , 2013 ) . `` Democratic
and Republican Ideologies Undergo Dramatic Role Reversal '' . HuffPost . Retrieved August 30 , 2017 .
</item 10>
=== END ITEM id=70a5f8d86cb84564 ===

=== ITEM id=0c7dcfeccbd186c9 ===
Gold sentences and facts:
Sentence 0: It was suggested that the interior of the 45 metre ( 147 ft ) wide dome should be covered with a mosaic decoration to make the most of the available light coming through the circular windows of the drum and through the lantern . Brunelleschi had proposed the vault to glimmer with resplendent gold , but his death in 1446 put an end to this project , and the walls of the dome were whitewashed . Grand Duke Cosimo I de ' Medici decided to have the dome painted with a representation of The Last Judgment . This enormous work , 3,600 metres2 ( 38 750 ft2 ) of painted surface , was started in 1568 by Giorgio Vasari and Federico Zuccari and would last till 1579 . The upper portion , near the lantern , representing The 24 Elders of Apoc. 4 was finished by Vasari before his death in 1574 . Federico Zuccari and a number of collaborators , such as Domenico Cresti , finished the other portions : ( from top to bottom ) Choirs of Angels ; Christ , Mary and Saints ; Virtues , Gifts of the Holy Spirit and Beatitudes ; and at the bottom of the cupola : Capital Sins and Hell . These frescoes are considered Zuccari 's greatest work . But the quality of the work is uneven because of the input of different artists and the different techniques . Vasari had used true fresco , while Zuccari had painted in secco . During the restoration work , which ended in 1995 , the entire pictorial cycle of The Last Judgment was photographed with specially designed equipment and all the information collected in a catalogue . All the restoration information along with reconstructed images of the frescos were stored and managed in the Thesaurus Florentinus computer system .
  Fact 0.0: It was suggested that the interior of the 45 metre (147 ft) wide dome should be covered with a mosaic decoration.
  Fact 0.1: The mosaic decoration was suggested to make the most of the available light coming through the circular windows of the drum.
  Fact 0.2: The mosaic decoration was suggested to make the most of the available light coming through the lantern.
  Fact 0.3: Brunelleschi had proposed that the vault glimmer with resplendent gold.
  Fact 0.4: Brunelleschi died in 1446.
  Fact 0.5: Brunelleschi's death in 1446 put an end to the gold vault project.
  Fact 0.6: The walls of the dome were whitewashed.
  Fact 0.7: Grand Duke Cosimo I de' Medici decided to have the dome painted with a representation of The Last Judgment.
  Fact 0.8: The painting of The Last Judgment covers 3,600 square metres (38,750 square feet) of painted surface.
  Fact 0.9: The painting of The Last Judgment was started in 1568 by Giorgio Vasari and Federico Zuccari.
  Fact 0.10: The painting of The Last Judgment would last until 1579.
  Fact 0.11: The upper portion of the painting, near the lantern, represents The 24 Elders of Apoc. 4.
  Fact 0.12: Vasari finished the upper portion of the painting before his death in 1574.
  Fact 0.13: Federico Zuccari and a number of collaborators, such as Domenico Cresti, finished the other portions of the painting.
  Fact 0.14: The other portions, from top to bottom, are Choirs of Angels; Christ, Mary and Saints; Virtues, Gifts of the Holy Spirit and Beatitudes; and at the bottom of the cupola, Capital Sins and Hell.
  Fact 0.15: The frescoes are considered Zuccari's greatest work.
  Fact 0.16: The quality of the work is uneven because of the input of different artists and the different techniques.
  Fact 0.17: Vasari had used true fresco.
  Fact 0.18: Zuccari had painted in secco.
  Fact 0.19: The restoration work ended in 1995.
  Fact 0.20: During the restoration work, the entire pictorial cycle of The Last Judgment was photographed with specially designed equipment.
  Fact 0.21: All the information collected during the restoration was put into a catalogue.
  Fact 0.22: All the restoration information, along with reconstructed images of the frescos, was stored and managed in the Thesaurus Florentinus computer system.

Retrieved items:
<item 1>
Florence Cathedral
Unsourced material may be challenged and removed . ( July 2012 ) ( Learn how and when to remove this template message ) Interior of the cathedral Huge clock
decorated by Paolo Uccello . Dante and the Divine Comedy . Trompe l'oeil of Niccolò da Tolentino . Vasari 's fresco begun in 1568 , and completed by Federico
Zuccari in 1579 . Detail under the dome . The Last Judgement ( detail ) under the dome . Tomb of Antonio d'Orso by Tino da Camaino . The Gothic interior is vast
and gives an empty impression . The relative bareness of the church corresponds with the austerity of religious life , as preached by Girolamo Savonarola . Many
decorations in the church have been lost in the course of time , or have been transferred to the Museum Opera del Duomo , such as the magnificent cantorial
pulpits ( the singing galleries for the choristers ) of Luca della Robbia and Donatello .
</item 1>

<item 2>
Florence Cathedral
Grand Duke Cosimo I de ' Medici decided to have the dome painted with a representation of The Last Judgment . This enormous work , 3,600 metres2 ( 38 750 ft2 )
of painted surface , was started in 1568 by Giorgio Vasari and Federico Zuccari and would last till 1579 . The upper portion , near the lantern , representing
The 24 Elders of Apoc. 4 was finished by Vasari before his death in 1574 . Federico Zuccari and a number of collaborators , such as Domenico Cresti , finished
the other portions : ( from top to bottom ) Choirs of Angels ; Christ , Mary and Saints ; Virtues , Gifts of the Holy Spirit and Beatitudes ; and at the bottom
of the cupola : Capital Sins and Hell . These frescoes are considered Zuccari 's greatest work . But the quality of the work is uneven because of the input of
different artists and the different techniques . Vasari had used true fresco , while Zuccari had painted in secco .
</item 2>

<item 3>
Florence Cathedral
The two main competitors were two master goldsmiths , Lorenzo Ghiberti and Filippo Brunelleschi , the latter of whom was supported by Cosimo de Medici .
Ghiberti had been the winner of a competition for a pair of bronze doors for the Baptistery in 1401 and lifelong competition between the two remained sharp .
Brunelleschi won and received the commission . Ghiberti , appointed coadjutator , drew a salary equal to Brunelleschi 's and , though neither was awarded the
announced prize of 200 florins , was promised equal credit , although he spent most of his time on other projects . When Brunelleschi became ill , or feigned
illness , the project was briefly in the hands of Ghiberti . But Ghiberti soon had to admit that the whole project was beyond him . In 1423 , Brunelleschi was
back in charge and took over sole responsibility . Work started on the dome in 1420 and was completed in 1436 . The cathedral was consecrated by Pope Eugene IV
on March 25 , 1436 , ( the first day of the year according to the Florentine calendar ) .
</item 3>

<item 4>
Graph query row:
person: Federico Zuccari
relationship: CREATED
work: The Last Judgment
date: None
</item 4>

<item 5>
Florence Cathedral
Jump up ^ As referenced in `` Cupola di Santa Maria del Fiore : il cantiere di restauro 1980 - 1995 '' , a cura di Cristina Acidini Luchinat e Riccardo Dalla
Negra published by Istituto Poligrafico e Zecca dello Stato ( Roma ) in 1995 ( ISBN 8824039561 ) Jump up ^ Thesaurus Florentinus project page ( in Italian ) ,
Soprintendenza ai Beni Architettonici e Paesaggisitici di Firenze , Ministero dei Beni Culturali References ( edit ) Bartlett , Kenneth R. ( 1992 ) . The
Civilization of the Italian Renaissance . Toronto : D.C. Heath and Company . ISBN 0 - 669 - 20900 - 7 ( Paperback ) . King , Ross ( 2000 ) . Brunelleschi 's
Dome : How a Renaissance Genius Reinvented Architecture . London : Chatto & Windus . Jepson , Tim ( 2001 ) . The National Geographic Traveler , Florence &
Tuscany . National Geographic Society .
</item 5>

<item 6>
Florence Cathedral
This fresco , transferred on canvas in the 19th century , in the same style as the previous one , is painted in a color resembling marble . However , it is more
richly decorated and gives more the impression of movement . Both frescoes portray the condottieri as heroic figures riding triumphantly . Both painters had
problems when applying in painting the new rules of perspective to foreshortening : they used two unifying points , one for the horse and one for the pedestal ,
instead a single unifying point . Busts of Giotto ( by Benedetto da Maiano ) , Brunelleschi ( by Buggiano - 1447 ) , Marsilio Ficino , and Antonio Squarcialupi
( a most famous organist ) . These busts all date from the 15th and the 16th centuries . Above the main door is the colossal clock face with fresco portraits of
four Prophets or Evangelists by Paolo Uccello ( 1443 ) . This one - handed liturgical clock shows the 24 hours of the hora italica ( Italian time ) , a period
of time ending with sunset at 24 hours .
</item 6>

<item 7>
Florence Cathedral
Vasari had used true fresco , while Zuccari had painted in secco . During the restoration work , which ended in 1995 , the entire pictorial cycle of The Last
Judgment was photographed with specially designed equipment and all the information collected in a catalogue . All the restoration information along with
reconstructed images of the frescos were stored and managed in the Thesaurus Florentinus computer system . Crypt ( edit ) Tomb of Filippo Brunelleschi . The
cathedral underwent difficult excavations between 1965 and 1974 . The archaeological history of this huge area was reconstructed through the work of Dr.
Franklin Toker : remains of Roman houses , an early Christian pavement , ruins of the former cathedral of Santa Reparata and successive enlargements of this
church . Close to the entrance , in the part of the crypt open to the public , is the tomb of Brunelleschi . While its location is prominent , the actual tomb
is simple and humble . That the architect was permitted such a prestigious burial place is proof of the high esteem he was given by the Florentines .
</item 7>

<item 8>
Florence Cathedral
Above this shrine is the painting Last Supper by the lesser - known Giovanni Balducci . There was also a glass - paste mosaic panel The Bust of Saint Zanobius
by the 16th - century miniaturist Monte di Giovanni , but it is now on display in the Museum Opera del Duomo . Many decorations date from the 16th - century
patronage of the Grand Dukes , such as the pavement in colored marble , attributed to Baccio d'Agnolo and Francesco da Sangallo ( 1520 -- 26 ) . Some pieces of
marble from the façade were used , topside down , in the flooring ( as was shown by the restoration of the floor after the 1966 flooding ) . It was suggested
that the interior of the 45 metre ( 147 ft ) wide dome should be covered with a mosaic decoration to make the most of the available light coming through the
circular windows of the drum and through the lantern . Brunelleschi had proposed the vault to glimmer with resplendent gold , but his death in 1446 put an end
to this project , and the walls of the dome were whitewashed .
</item 8>

<item 9>
Graph query row:
person: Giorgio Vasari
relationship: CREATED
work: The Last Judgment
date: None
</item 9>

<item 10>
Florence Cathedral
As this cathedral was built with funds from the public , some important works of art in this church honour illustrious men and military leaders of Florence :
Lorenzo Ghiberti had a large artistic impact on the cathedral . Ghiberti worked with Filippo Brunelleschi on the cathedral for eighteen years and had a large
number of projects on almost the whole east end . Some of his works were the stained glass designs , the bronze shrine of Saint Zenobius and marble revetments
on the outside of the cathedral . Dante Before the City of Florence by Domenico di Michelino ( 1465 ) . This painting is especially interesting because it shows
us , apart from scenes of the Divine Comedy , a view on Florence in 1465 , a Florence such as Dante himself could not have seen in his time . Funerary Monument
to Sir John Hawkwood by Paolo Uccello ( 1436 ) . This almost monochrome fresco , transferred to canvas in the 19th century , is painted in terra verde , a color
closest to the patina of bronze . Equestrian statue of Niccolò da Tolentino by Andrea del Castagno ( 1456 ) .
</item 10>
=== END ITEM id=0c7dcfeccbd186c9 ===

=== ITEM id=bf0355b479b7e9fe ===
Gold sentences and facts:
Sentence 0: A noteworthy example of a trabeated system is in Volubilis , from the Roman era , where one side of the Decumanus Maximus is lined with trabeated elements , while the opposite side of the roadway is designed in arched style .
  Fact 0.0: A noteworthy example of a trabeated system is in Volubilis.
  Fact 0.1: The Volubilis trabeated example is from the Roman era.
  Fact 0.2: In Volubilis, one side of the Decumanus Maximus is lined with trabeated elements.
  Fact 0.3: In Volubilis, the opposite side of the roadway is designed in arched style.

Retrieved items:
<item 1>
Romanesque art
Of all the many examples that occur on Romanesque portals , one of the most outstanding is that of the central figure of Christ at La Madeleine , Vezelay .
Another influence from Insular art are engaged and entwined animals , often used to superb effect in capitals ( as at Silos ) and sometimes on a column itself (
as at Moissac ) . Much of the treatment of paired , confronted and entwined animals in Romanesque decoration has similar Insular origins , as do animals whose
bodies tail into purely decorative shapes . ( Despite the adoption of Hiberno - Saxon traditions into Romanesque styles in England and on the continent , the
influence was primarily one - way . Irish art during this period remained isolated , developing a unique amalgam of native Irish and Viking styles which would
be slowly extinguished and replaced by mainstream Romanesque style in the early 13th century following the Anglo - Norman invasion of Ireland . ) Judas Iscariot
hangs himself , assisted by devils , always a favourite subject of carvers . Autun Cathedral Subject matter ( edit ) Most Romanesque sculpture is pictorial and
biblical in subject .
</item 1>

<item 2>
Bed
A mortise or slot is cut vertically in the bedpost . Pins are inserted horizontally in the bed post so that the pins perpendicularly intersect the mortise . For
example , if one looked in the mortise , one might see part of one horizontal pin at the bottom of the mortise and a part of a second pin toward the top of the
mortise . Hooks are installed at the end of the rail . Usually these hooks are part of a plate that is attached to the rail . The hooks then are inserted into
the bed post mortise and hook over the pins . Plate - and - hook fastener . Instead of pins inserted horizontally into the bedpost , an eye plate ( post plate )
is installed on the bedpost . The hooks are installed on the rail , either as surface mount or recessed . Depending on the hardware , the bedpost may require a
mortise in order to allow the hooks to fasten to the plate . This is also referred to as a keyhole fastener , especially if the connector is more of a `` plug
'' than a `` hook '' . Bed bolts ( `` through - bolts '' ) are a different means of a knock - down connection .
</item 2>

<item 3>
Post and lintel
See also ( edit ) Wikimedia Commons has media related to Post and Beam and Lintels . Architrave - structural lintel or beam resting on columns - pillars
Atalburu - Basque decorative lintel Dolmen - Neolithic megalithic tombs with structural stone lintels Dougong - traditional Chinese structural element I - beam
- steel lintels and beams Marriage stone - decorative lintel Timber framing - post and beam systems Structural design Opus caementicium Notes ( edit ) Jump up ^
Post and lintel is the main structural system in Northern China , the southern traditional timber buildings which use a column - and - tie structural system .
`` Structural Mechanism Of Southern Chinese Traditional Timber Frame Buildings '' SCIENCE CHINA Technological Sciences. 2011 , Vol 54 ( 7 )
http://www.redorbit.com/news/science/1112427956/structural-mechanism-of-southern-chinese-traditional-timber-frame-buildings/ Jump up ^ Summerson , 13 - 14 Jump
up ^ Summerson , 19 - 21 Jump up ^ C. Michael Hogan , Volubilis , Megalithic Portal , ed .
</item 3>

<item 4>
Post and lintel
The trabeated system is a fundamental principle of Neolithic architecture , Ancient Indian architecture , Ancient Greek architecture and Ancient Egyptian
architecture . Other trabeated styles are the Persian , Lycian , Japanese , traditional Chinese , and ancient Chinese architecture , especially in northern
China , and nearly all the Indian styles . The traditions are represented in North and Central America by Mayan architecture , and in South America by Inca
architecture . In all or most of these traditions , certainly in Greece and India , the earliest versions developed using wood , which were later translated
into stone for larger and grander buildings . Timber framing , also using trusses , remains common for smaller buildings such as houses to the modern day . The
biggest disadvantage to a post and lintel construction is the limited weight that can be held up , and the small distances required between the posts . Ancient
Roman architecture 's development of the arch allowed for much larger structures to be constructed . The arcuated system spreads larger loads more effectively ,
and replaced the post and lintel trabeated system in most larger buildings and structures , until the introduction of steel girder beams in the industrial era .
</item 4>

<item 5>
Post and lintel
Engineering ( edit ) Post and lintel construction is one of four ancient structural methods of building , the others being the corbel , arch - and - vault , and
truss . There are two main force vectors acting upon the post and lintel system : weight carrying compression at the joint between lintel and post , and tension
induced by deformation of self - weight and the load above between the posts . The two posts are under compression from the weight of the lintel ( or beam )
above . The lintel will deform by sagging in the middle because the underside is under tension and the topside is under compression . See also ( edit )
Wikimedia Commons has media related to Post and Beam and Lintels .
</item 5>

<item 6>
Florence Cathedral
The rails were connected end - to - end with lead - glazed iron splices . The cross ties and rails were notched together and then covered with the bricks and
mortar of the inner dome . The cross ties of the bottom chain can be seen protruding from the drum at the base of the dome . The others are hidden . Each stone
chain was supposed to be reinforced with a standard iron chain made of interlocking links , but a magnetic survey conducted in the 1970s failed to detect any
evidence of iron chains , which if they exist are deeply embedded in the thick masonry walls . Brunelleschi also included vertical `` ribs '' set on the corners
of the octagon , curving towards the center point . The Ribs , 13 feet ( 4 meters ) deep , are supported by 16 concealed ribs radiating from center . The ribs
had slits to take beams that supported platforms , thus allowing the work to progress upward without the need for scaffolding . A circular masonry dome can be
built without supports , called centering , because each course of bricks is a horizontal arch that resists compression .
</item 6>

<item 7>
Post and lintel
As with the Roman temple portico front and its descendents in later classical architecture , trabeated features were often retained in parts of buildings as an
aesthetic choice . The classical orders of Greek origin were in particular retained in buildings designed to impress , even though they usually had little or no
structural role . Contents ( hide ) 1 Lintel beams 2 Engineering 3 See also 4 Notes 5 References Lintel beams ( edit ) Further information : Beam ( structure )
In architecture , a post-and - lintel or trabeated system refers to the use of horizontal beams or lintels which are borne up by columns or posts . The name is
from the Latin trabs , beam ; influenced by trabeatus , clothed in the trabea , a ritual garment . A noteworthy example of a trabeated system is in Volubilis ,
from the Roman era , where one side of the Decumanus Maximus is lined with trabeated elements , while the opposite side of the roadway is designed in arched
style . In India the style was used originally for wooden construction , but later the technique was adopted for stone structures for decorated load - bearing
and purely ornamented non-structural purposes .
</item 7>

<item 8>
Romanesque art
The Gloucester candlestick , early 12th century The bronze Gloucester candlestick and the brass font of 1108 -- 17 now in Liège are superb examples , very
different in style , of metal casting . The former is highly intricate and energetic , drawing on manuscript painting , while the font shows the Mosan style at
its most classical and majestic . The bronze doors , a triumphal column and other fittings at Hildesheim Cathedral , the Gniezno Doors , and the doors of the
Basilica di San Zeno in Verona are other substantial survivals . The aquamanile , a container for water to wash with , appears to have been introduced to Europe
in the 11th century . Artisans often gave the pieces fantastic zoomorphic forms ; surviving examples are mostly in brass . Many wax impressions from impressive
seals survive on charters and documents , although Romanesque coins are generally not of great aesthetic interest . The Cloisters Cross is an unusually large
ivory crucifix , with complex carving including many figures of prophets and others , which has been attributed to one of the relatively few artists whose name
is known , Master Hugo , who also illuminated manuscripts .
</item 8>

<item 9>
Post and lintel
Post and lintel - wikipedia Post and lintel Jump to : navigation , search Post and lintel construction of the World Heritage Monument site Airavatesvara Temple
, India Leinster House in Dublin retains columns under a pediment for aesthetic reasons ; the similar façade at the White House at least holds up a porch Timber
frame of a modern building under construction , also using trusses Stonehenge , an example of Neolithic architecture post and lintel construction . In
architecture , post and lintel ( also called prop and lintel or a trabeated system ) is a building system where strong horizontal elements are held up by strong
vertical elements with large spaces between them . This is usually used to hold up a roof , creating a largely open space beneath , for whatever use the
building is designed for . The horizontal elements are called by a variety of names including lintel , header , architrave or beam , and the supporting vertical
elements may be called columns , pillars , or posts . The use of wider elements at the top of the post , called capitals , to help spread the load , is common
to many traditions .
</item 9>

<item 10>
Post and lintel
A. Burnham ( 2007 ) Jump up ^ L. Sprague De Camp , Ancient Engineers : Technology & Invention from the Earliest Times to the Renaissance ( U.S.A. : Barnes and
Noble , 1993 edition ) , 35 . References ( edit ) Summerson , John , The Classical Language of Architecture , 1980 edition , Thames and Hudson World of Art
series , ISBN 0500201773 Retrieved from `` https://en.wikipedia.org/w/index.php?title=Post_and_lintel&oldid=787548858 '' Categories : Architectural elements
Ancient Roman architectural elements Building Construction terminology Doors Windows Timber framing Structural system Talk Contents About Wikipedia فارسی
Gaeilge Edit links This page was last edited on 26 June 2017 , at 00 : 54 . Text is available under the Creative Commons Attribution - ShareAlike License ;
additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia
Foundation , Inc. , a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=bf0355b479b7e9fe ===
