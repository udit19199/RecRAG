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

This file has 12 items.

=== ITEM id=507637acc0d73a22 ===
Gold sentences and facts:
Sentence 0:  They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
  Fact 0.0: The unnamed subject competed at the 2015 Pacific Games
  Fact 0.1: The unnamed subject finished the 2015 Pacific Games with an 0-4 record
Sentence 1: Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
  Fact 1.0: Basketball at the 2015 Pacific Games was held in Port Moresby, Papua New Guinea
  Fact 1.1: Basketball at the 2015 Pacific Games was held at the BSP Arena and PNG Power Dome
  Fact 1.2: Basketball at the 2015 Pacific Games was held on 3 to 12 July 2015

Retrieved items:
<item 1>
Kiribati at the 2015 Pacific Games
Kiribati competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 A total of 86 competitors for Kiribati were listed as of 4 July 2015.
</item 1>

<item 2>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 2>

<item 3>
Graph query row:
event: 2015 Pacific Games
event_location: Port Moresby
place: Port Moresby
located_in: Papua New Guinea
participating_team: None
participation_relationship: None
participated: False
</item 3>

<item 4>
Men's football at the 2015 Pacific Games
The 2015 Pacific Games men's football tournament was the 14th edition of the Pacific Games men's football tournament.

 The men's football tournament was held in Port Moresby, Papua New Guinea between 3–17 July 2015 as part of the 2015 Pacific Games.

 The tournament was age-restricted and open to men's under-23 national teams only.
</item 4>

<item 5>
2015 Pacific Games
The 2015 Pacific Games was held in Port Moresby, Papua New Guinea, from 4 to 18 July 2015.

 It was the fifteenth staging of the Pacific Games.
</item 5>

<item 6>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 6>

<item 7>
Kiribati national basketball team
The Kiribati national basketball team are the basketball side that represent Kiribati in international competitions.

 They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
</item 7>

<item 8>
Powerlifting at the 2015 Pacific Games
Powerlifting at the 2015 Pacific Games was held from 9–10 July 2015 in the Sir John Guise Indoor Power Dome at Port Moresby, Papua New Guinea.

 The host nation Papua New Guinea was the dominant team, particularly in the women's divisions where it claimed six of the seven possible gold medals.

 Nauru was the strongest team in the men's divisions, winning four of the eight weight categories.

 Telupe Iosefa received the first ever gold medal won by Tuvalu at the Pacific Games in the 120 kg male division.
</item 8>

<item 9>
Basketball at the 2015 Pacific Games
Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
</item 9>

<item 10>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 10>
=== END ITEM id=507637acc0d73a22 ===

=== ITEM id=eb949541e34b67c7 ===
Gold sentences and facts:
Sentence 0: Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct "Final Destination 5" six years later, and filmed in the IMAX 3D format.
  Fact 0.0: Aliens of the Deep is a 2005 documentary film.
  Fact 0.1: Aliens of the Deep was directed in part by James Cameron.
  Fact 0.2: James Cameron directed Aliens of the Deep alongside fellow cameraman and friend Steven Quale.
  Fact 0.3: Steven Quale would go on to direct Final Destination 5 six years later.
  Fact 0.4: Aliens of the Deep was filmed in the IMAX 3D format.
Sentence 1:  Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of the planet's more unusual forms of life.
  Fact 1.0: James Cameron teams with NASA scientists to explore the Mid-Ocean Ridges.
  Fact 1.1: The Mid-Ocean Ridges are submerged chains of mountains.
  Fact 1.2: The Mid-Ocean Ridges are in the Atlantic and Pacific oceans.
  Fact 1.3: The Mid-Ocean Ridges are home to some of the planet's more unusual forms of life.
Sentence 2: Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.
  Fact 2.0: Poetry in Motion is a 1982 Canadian documentary film.
  Fact 2.1: Poetry in Motion was directed by Ron Mann.
  Fact 2.2: Poetry in Motion features contemporary North American poetry.
  Fact 2.3: Poetry in Motion features contemporary North American music.
Sentence 3:  Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.
  Fact 3.0: Poetry in Motion features some Black Mountain poets.
  Fact 3.1: Poetry in Motion features some Beat poets.
  Fact 3.2: Poetry in Motion features some minimalist poets.
  Fact 3.3: Poetry in Motion features some avant-garde poets.
Sentence 4:  It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.
  Fact 4.0: Poetry in Motion was released in theaters.
  Fact 4.1: Poetry in Motion was later distributed on VHS.
  Fact 4.2: Poetry in Motion was later distributed on LaserDisc.
  Fact 4.3: Poetry in Motion was later distributed on DVD.
Sentence 5:  An extended CD-ROM version was also released.
  Fact 5.0: An extended CD-ROM version of Poetry in Motion was also released.

Retrieved items:
<item 1>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 1>

<item 2>
SlamNation
SlamNation is a documentary film by director Paul Devlin.

 The film follows the National Poetry Slam in Portland, Oregon.
</item 2>

<item 3>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 3>

<item 4>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 4>

<item 5>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 5>

<item 6>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 6>

<item 7>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 7>

<item 8>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 8>

<item 9>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 9>

<item 10>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 10>
=== END ITEM id=eb949541e34b67c7 ===

=== ITEM id=dd1acd0d4b8541e0 ===
Gold sentences and facts:
Sentence 0: The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver Division, the Shenango Division, and the Conneaut Division.
  Fact 0.0: The Beaver and Erie Canal was also known as the Erie Extension Canal.
  Fact 0.1: The Beaver and Erie Canal was part of the Pennsylvania Canal system.
  Fact 0.2: The Beaver and Erie Canal consisted of three sections.
  Fact 0.3: The three sections of the Beaver and Erie Canal were the Beaver Division, the Shenango Division, and the Conneaut Division.
Sentence 1: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.
  Fact 1.0: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp.
  Fact 1.1: The Dismal Swamp Canal is located in Virginia and North Carolina.
  Fact 1.2: Virginia and North Carolina are in the United States.

Retrieved items:
<item 1>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 1>

<item 2>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 2>

<item 3>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 3>

<item 4>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 4>

<item 5>
Great Dismal Swamp National Wildlife Refuge
The Great Dismal Swamp National Wildlife Refuge was created in 1974 to help protect and preserve a portion of the Great Dismal Swamp, a marshy region on the
Coastal Plain of southeastern Virginia and northeastern North Carolina between Norfolk, Virginia, and Elizabeth City, North Carolina in the United States.

 It is located in parts of the independent cities of Chesapeake and Suffolk in Virginia, and the counties of Camden, Gates, and Pasquotank in North Carolina.
</item 5>

<item 6>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 6>

<item 7>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 7>

<item 8>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 8>

<item 9>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 9>

<item 10>
Dismal Swamp State Park
Dismal Swamp State Park is a North Carolina state park in Camden County, North Carolina in the United States.

 The park was created as a state natural area in 1974 with the help of The Nature Conservancy, and on July 28, 2007 the NC General Assembly re-designated it as
a state park.

 It opened to the public in 2008.

 This marked the first time that public access to Great Dismal Swamp was made possible in North Carolina.

 The park covers 14432 acre of protected land on the North Carolina/Virginia border.

 Park offices are three miles (5 km) south of the border on U.S. Route 17 near South Mills.

 Features of the park include the canal which is used regularly by boaters using the Intracoastal Waterway and several miles of hiking and biking trails.
</item 10>
=== END ITEM id=dd1acd0d4b8541e0 ===

=== ITEM id=b44eea7dfee99c2b ===
Gold sentences and facts:
Sentence 0:  He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.
  Fact 0.0: This person creates bespoke millinery under the J Smith Esquire brand.
  Fact 0.1: This person creates bespoke millinery for a portfolio of private clients.
  Fact 0.2: The portfolio of private clients includes Angelina Jolie.
  Fact 0.3: The portfolio of private clients includes Brad Pitt.
  Fact 0.4: The portfolio of private clients includes Amal Clooney.
Sentence 1: Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and human rights.
  Fact 1.0: Amal Clooney is also known by the name Alamuddin (née Alamuddin).
  Fact 1.1: Amal Clooney's Arabic name is أمل علم الدين.
  Fact 1.2: Amal Clooney was born on 3 February 1978.
  Fact 1.3: Amal Clooney is a barrister at Doughty Street Chambers.
  Fact 1.4: Amal Clooney specialises in international law.
  Fact 1.5: Amal Clooney specialises in human rights.

Retrieved items:
<item 1>
Doughty Street Chambers
Doughty Street Chambers is a British set of barristers' chambers situated in Bristol, Manchester and London's Doughty Street, undertaking criminal justice,
public law, immigration, employment, human rights and civil liberties work.
</item 1>

<item 2>
Amal Clooney
Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and
human rights.

 Her clients include Julian Assange, the founder of WikiLeaks, in his fight against extradition.

 She has also represented the former prime minister of Ukraine, Yulia Tymoshenko, and Egyptian-Canadian journalist Mohamed Fahmy.

 She is married to the American actor George Clooney.
</item 2>

<item 3>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 3>

<item 4>
Pupil master
Occasionally, a pupil barrister may undertake a third "six", extending the training period a further six months.

 At the end of pupillage, to continue practising the law, a barrister may attempt to become a tenant in a set of barristers' chambers, or find a position as an
employed barrister.
</item 4>

<item 5>
Mónica Feria Tinta
Mónica Feria Tinta is a leading public international lawyer.

 She practises as a Barrister at 20 Essex Street Chambers.

 In 2000 she became the first Peruvian-born lawyer to receive the Diploma of the Hague Academy of International Law in history.

 Her litigation work led to the first international human rights court decision ordering the prosecution of a former Head of State for crimes under
international law.

 She was awarded the Inge Genefke International Award for her work as an international lawyer in 2006 and was co-recipient of the Gruber Justice Prize 2007, for
her contributions advancing the cause of justice as delivered through the legal system.

 She is a member of the Bar of England and Wales, and the American Society of International Law.
</item 5>

<item 6>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 6>

<item 7>
Edmund Bartley-Denniss
Sir Edmund Robert Bartley Bartley-Denniss KC (9 April 1854 – 20 March 1931), born Edmund Robert Bartley Denniss, was a barrister, prominent Freemason and
Conservative Party Member of Parliament in the United Kingdom.

 He was also a pioneer of the sport of cycling in Britain.
</item 7>

<item 8>
Shahram Taghavi
Shahram Taghavi is a barrister practising in the United Kingdom.

 He is a specialist in Judicial Review, Human Rights, Immigration and EU law.

 He was called to the England & Wales bar in 1994.

 He is a partner at Article 1, a law firm specialising in immigration, EU freedoms of movement and human rights law.

 He practised as an independent barrister at Doughty Street Chambers specialising in immigration and public law.

 He was head of both the Immigration and Public Law departments at Simons Muirhead & Burton solicitors.

 He was a Senior Barrister at Bates Wells & Braithwaite LLP as a member of their Public & Regulatory and Immigration departments, Deputy Head of Immigration at
Lewis Silkin LLP, Head of Immigration at Charles Russell LLP (now Charles Russell Speechly LLP) and National Head of Immigration at Grant Thornton LLP.
</item 8>

<item 9>
Barbados Bar Association
The Barbados Bar Association is a voluntary association of attorneys in Barbados who practise at the independent bar as barristers and Queen's Counsel.

 It was created by the Barbados Bar Association Act of 1940.
</item 9>

<item 10>
Pupil master
A pupil master ('pupilmaster' or 'pupil-master') or in the case of a female barrister, 'pupil mistress' etc., is the former name given to an experienced
barrister who a pupil shadows during their pupillage.

 The term pupil master or mistress has now been replaced by the term 'pupil supervisor'.

 Barristers are called to the Bar via one of the four Inns of Court upon successful completion of the BPTC and having undertaken a required number of
"qualifying sessions" in their chosen Inn of Court.

 In most cases, the newly called barrister is then required to undertake training for a period of at least a year before the barrister can start their own
private practice.

 This training period is known as pupillage, usually split into two periods of six months known as "sixes".

 The first "six" is a non-practising six, during which the pupil will shadow their pupil master; the second is usually a practising "six", when the pupil, with
their pupil master's permission, can undertake the supply of legal services and exercise rights of audience in court.
</item 10>
=== END ITEM id=b44eea7dfee99c2b ===

=== ITEM id=55988be4c289cf32 ===
Gold sentences and facts:
Sentence 0: Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.
  Fact 0.0: Bernard Malamud was born on April 26, 1914.
  Fact 0.1: Bernard Malamud died on March 18, 1986.
  Fact 0.2: Bernard Malamud was an American novelist.
  Fact 0.3: Bernard Malamud was an American short story writer.
Sentence 1: Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.
  Fact 1.0: Lois McMaster Bujold was born on November 2, 1949.
  Fact 1.1: Lois McMaster Bujold is an American speculative fiction writer.

Retrieved items:
<item 1>
Horizon (novel)
Horizon is a fantasy novel by American writer Lois McMaster Bujold.

 It is the fourth in the tetralogy "The Sharing Knife".
</item 1>

<item 2>
Shards of Honor
Shards of Honor is an English language science fiction novel by Lois McMaster Bujold, first published in June 1986.

 It is a part of the Vorkosigan Saga, and is the first full-length novel in publication order.

 "Shards of Honor" is paired with Bujold's 1991 "Barrayar" in the omnibus "Cordelia's Honor" (1996).
</item 2>

<item 3>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 3>

<item 4>
Lois McMaster Bujold
Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.

 She is one of the most acclaimed writers in her field, having won the Hugo Award for best novel four times, matching Robert A. Heinlein's record, not counting
his Retro Hugo.

 Her novella "The Mountains of Mourning" won both the Hugo Award and Nebula Award.

 In the fantasy genre, "The Curse of Chalion" won the Mythopoeic Award for Adult Literature and was nominated for the 2002 World Fantasy Award for best novel,
and both her fourth Hugo Award and second Nebula Award were for "Paladin of Souls".

 In 2011 she was awarded the Skylark Award.

 In 2013 she was awarded the Forry Award.

 In 2017 she won a Hugo Award for Best Series, for the Vorkosigan Saga.
</item 4>

<item 5>
Passage (Bujold novel)
Passage is a novel by American writer Lois McMaster Bujold, published in 2008.

 It is the third in the tetralogy "The Sharing Knife".
</item 5>

<item 6>
Cryoburn
Cryoburn is a science fiction novel by Lois McMaster Bujold, first published in October 2010.

 Part of the Vorkosigan Saga, it was nominated for the Hugo Award for Best Novel in 2011, as Bujold's ninth Best Novel nomination.

 Also in 2011, it was one of the top five finishers in the poll for the Locus Award for Best Science Fiction Novel.
</item 6>

<item 7>
Barrayar
Barrayar is a science fiction novel by Lois McMaster Bujold.

 It was first published as four installments in "Analog" in July–October 1991, and then published in book form by Baen Books in October 1991.

 "Barrayar" won both the Hugo Award for Best Novel and the Locus Award for Best Science Fiction Novel in 1992.

 It is a part of the Vorkosigan Saga, and is the seventh full-length novel of the series, in publication order.

 "Barrayar" is a direct sequel to Bujold's first novel, "Shards of Honor" (1986), and the two are paired in the 1996 omnibus "Cordelia's Honor".
</item 7>

<item 8>
Lois McMaster Bujold (born 1949), acclaimed American speculative-fiction writer, created the Vorkosigan Saga and won four Hugo Awards for Best Novel; the series
won Best Series in 2017. Its works include *Shards of Honor* (1986), followed by Hugo- and Locus-winning *Barrayar* (1991); *Ethan of Athos* (1986), featuring
Dr. Ethan Urquhart; *Brothers in Arms* (
</item 8>

<item 9>
Gentleman Jole and the Red Queen
Gentleman Jole and the Red Queen is a science fiction novel by Lois McMaster Bujold.

 It is an installment in Bujold's Vorkosigan Saga.

 Bujold has described it as "not a war story.

 It is about grownups."
</item 9>

<item 10>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 10>
=== END ITEM id=55988be4c289cf32 ===

=== ITEM id=31e8ea474437dca5 ===
Gold sentences and facts:
Sentence 0: Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.
  Fact 0.0: Alice Ann Munro's nee name is Laidlaw.
  Fact 0.1: Alice Ann Munro was born on 10 July 1931.
  Fact 0.2: Alice Ann Munro is a Canadian short story writer.
  Fact 0.3: Alice Ann Munro is a Nobel Prize winner.
Sentence 1: Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well as her left-wing sympathies and political activism.
  Fact 1.0: Lillian Florence Hellman was born on June 20, 1905.
  Fact 1.1: Lillian Florence Hellman died on June 30, 1984.
  Fact 1.2: Lillian Florence Hellman was an American dramatist.
  Fact 1.3: Lillian Florence Hellman was an American screenwriter.
  Fact 1.4: Lillian Florence Hellman is known for her success as a playwright on Broadway.
  Fact 1.5: Lillian Florence Hellman is known for her left-wing sympathies.
  Fact 1.6: Lillian Florence Hellman is known for her political activism.

Retrieved items:
<item 1>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 1>

<item 2>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 2>

<item 3>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 3>

<item 4>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 4>

<item 5>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 5>

<item 6>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 6>

<item 7>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 7>

<item 8>
Anne Perry
Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk
series.

 At the age of fifteen she was convicted of participating in the murder of her friend's mother, in 1954.

 She changed her name after serving her five-year sentence.
</item 8>

<item 9>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 9>

<item 10>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 10>
=== END ITEM id=31e8ea474437dca5 ===

=== ITEM id=579acb24b110e668 ===
Gold sentences and facts:
Sentence 0: Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming pieces and a game system.
  Fact 0.0: Icehouse pieces are also called Icehouse Pyramids.
  Fact 0.1: Treehouse pieces are also called Treehouse Pyramids.
  Fact 0.2: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, and Treehouse Pyramids are officially called Looney Pyramids.
  Fact 0.3: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are nestable pyramid-shaped gaming pieces.
  Fact 0.4: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are stackable pyramid-shaped gaming pieces.
  Fact 0.5: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are a game system.
Sentence 1:  The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
  Fact 1.0: The Looney Pyramids game system was invented by Andrew Looney.
  Fact 1.1: The Looney Pyramids game system was invented by John Cooper.
  Fact 1.2: The Looney Pyramids game system was invented in 1987.
  Fact 1.3: The Looney Pyramids game system was originally for use in the game of Icehouse.
Sentence 2:  In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
  Fact 2.0: "Kill Doctor Lucky" won the Origins Award in 1998.
  Fact 2.1: "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".

Retrieved items:
<item 1>
Can't Help Myself
"Can't Help Myself" is the first single released by the Australian synthpop/rock band Flowers, later known as Icehouse.

 It was released in May 1980 as a 7" vinyl single on independent label, Regular Records, five months ahead of debut album "Icehouse".

 A 10" vinyl single was released in July and had a cover depicting individual images of band members diagonally across the band's name and the single's title
(see infobox at right middle).

 It peaked at #10 on the Australian Singles Charts.
</item 1>

<item 2>
Icehouse pieces
Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming
pieces and a game system.

 The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
</item 2>

<item 3>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 3>

<item 4>
The Hunter (EP)
The Hunter is the first EP (and follow up to the 2010 solo album "The Boxer") by Kele Okereke (under the professional name Kele), lead singer and rhythm
guitarist of the British rock band Bloc Party.

 It was released on 7 November 2011 by Wichita Recordings in the UK, set back a week from the original release date, and 3 days earlier on 4 November 2011 by
Wichita Recordings and Liberator Music in Australia.

 The first single released from the EP was "What Did I Do?

", which features guest vocals from Lucy Taylor.

 The music video was released on 13 September 2011, with the song released as a digital single in Japan 10 days later.
</item 4>

<item 5>
Icehouse (song)
"Icehouse" is a song by the Australian rock band Flowers.

 It was released as a single in Europe in 1982 by Chrysalis Records from the band's first album, "Icehouse", after the band changed its name to Icehouse.

 In the United States, the song peaked at number 28 on the "Billboard" Top Tracks chart in 1981.
</item 5>

<item 6>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 6>

<item 7>
No Promises (Icehouse song)
"No Promises" is the first single released by Australian band, Icehouse from the band's 1986 album, "Measure for Measure" and was released in November 1985 on
Regular Records as 7" Vinyl Single and 12" Vinyl Single formats.

 It was also released in the UK and Europe by Chrysalis Records again on 7" and 12" formats but with different track listings.

 It was subsequently released in the US by Chrysalis on 7" and 12" formats, again with different track listings.

 It peaked at #30 on the Australian singles charts.

 A remix version by [love] tatto was released on the Icehouse album "Meltdown" in 2002.
</item 7>

<item 8>
Treehouse (game)
Treehouse is a game in which players try to get their configuration of Icehouse pieces to match the central configuration, shared by all players.

 The rolling of the special "Treehouse Die" tells the player what kind of move to make to change his own or the central configuration, and then he does so to
best move towards the goal.
</item 8>

<item 9>
Boxes (Icehouse album)
Boxes is a soundtrack album by Australian band Icehouse, released by Festival Records / Chrysalis Records in November 1985.

 The work was originally conceived by its composers, Iva Davies and Robert Kretschmer, in collaboration with choreographer Graeme Murphy of the Sydney Dance
Company for performance as the ballet "Boxes".

 The first live performance of "Boxes" was given by the Sydney Dance Company together with Davies and Kretschmer of Icehouse and guest percussionist Masaki
Tanazawa in the Opera Theatre of the Sydney Opera House on 7 November 1985.

 "Boxes" was released in the US under the name Sydney Dance Company.

 The single "No Promises" was released as a 12" in November and a cassingle in December on Regular Records for Australian and New Zealand markets, it contained
other tracks from "Boxes"; two tracks from "Boxes" were later released on their next studio album "Measure for Measure" in April 1986.
</item 9>

<item 10>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 10>
=== END ITEM id=579acb24b110e668 ===

=== ITEM id=87bb08f2c9a65cf5 ===
Gold sentences and facts:
Sentence 0:  This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's "Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
  Fact 0.0: This episode is a milestone in the history of the series.
  Fact 0.1: In this episode, Snowball II is killed off.
  Fact 0.2: Homer's mother died in season 19's Mona Leaves-a.
  Fact 0.3: Excluding the death of Homer's mother, the killing of Snowball II is to date the closest thing to an actual Simpson family member being killed off.
Sentence 1: Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".
  Fact 1.0: Mona Penelope Simpson's maiden name is Olsen.
  Fact 1.1: Mona Penelope Simpson is a recurring fictional character.
  Fact 1.2: Mona Penelope Simpson is a character in the animated television series The Simpsons.

Retrieved items:
<item 1>
My Mother the Carjacker
"My Mother the Carjacker" is the second episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It first aired on the Fox network in the United States on November 9, 2003.

 Homer receives a cryptic message in the newspaper informing him to come to a certain place at midnight, and soon discovers that the person who wrote the
message is his mother, Mona Simpson.

 It was written by Michael Price and directed by Nancy Kruse.

 Glenn Close makes her second of six guest spots as Homer's mother.

 It has a direct link from the season seven episode, "Mother Simpson".

 It was nominated for a Writers Guild of America Award in 2004.

 In its original run, the episode received 12.4 million viewers.
</item 1>

<item 2>
Future-Drama
"Future-Drama" is the fifteenth episode of "The Simpsons"<nowiki>'</nowiki> sixteenth season.

 The 350th episode overall, it originally aired on the Fox network in the United States on April 17, 2005.

 In the episode, Bart and Lisa stumble into Professor Frink's basement, and he gives them a look into their future as teenagers as they get ready for their high
school graduation.

 Matt Selman wrote the episode, and Mike B. Anderson served as director.

 Amy Poehler and John DiMaggio guest-starred as the characters of Jenda and Bender respectively.
</item 2>

<item 3>
Good Night (The Simpsons short)
"Good Night" (also known as "Good Night Simpsons") is the first of forty-eight Simpsons shorts that appeared on the variety show "The Tracey Ullman Show".

 It originally aired on the Fox network in the United States on April 19, 1987, during the third episode of "The Tracey Ullman Show" and marks the first
appearance of the Simpson family — Homer, Marge, Bart, Lisa, and Maggie — on television.

 After three seasons on Tracey Ullman, the shorts would be adapted into the animated show "The Simpsons".

 "Good Night" has since been aired on the show in the episode "The Simpsons 138th Episode Spectacular" (in its entirety), along with several other Ullman
shorts, and is one of the few shorts to ever be released on DVD, being included in the Season 1 DVD set.
</item 3>

<item 4>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 4>

<item 5>
Judge Me Tender
"Judge Me Tender" is the twenty-third episode and season finale of "The Simpsons'" twenty-first season.

 The 464th episode of the series overall, it originally aired on the Fox network in the United States on May 23, 2010.

 In the episode, Moe discovers his talent for judging in competitions and is invited to appear on the show "American Idol".

 Meanwhile, Homer drives Marge crazy when he starts spending too much time at home, and Lisa tries to comfort Santa's Little Helper.
</item 5>

<item 6>
I, (Annoyed Grunt)-bot
"I, (Annoyed Grunt)-Bot", also known as ""I, D'oh-Bot"", is the ninth episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It originally aired on the Fox network in the United States on January 11, 2004.

 This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's
"Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
</item 6>

<item 7>
Lisa the Drama Queen
"Lisa the Drama Queen" is the ninth episode of the twentieth season of "The Simpsons".

 It originally aired on the Fox network in the United States on January 25, 2009 and guest starred Emily Blunt as Juliet.

 A special version of the end credits theme was performed by Fall Out Boy, although they do not guest star in the actual episode.

 This was the last episode in the series to be presented only in standard definition, the first regular episode to begin right after the opening credits without
a commercial break, with an episode now having four segments, and the last episode to use the original opening sequence starting from Season 2.

 Brian Kelley returns as a writer after five seasons of being absent from the show.

 The episode is very loosely based on the 1994 film "Heavenly Creatures".
</item 7>

<item 8>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 8>

<item 9>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 9>

<item 10>
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 10>
=== END ITEM id=87bb08f2c9a65cf5 ===

=== ITEM id=fc9f0b46b6097fb1 ===
Gold sentences and facts:
Sentence 0:  It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts Mika Brzezinski and Willie Geist.
  Fact 0.0: The program described features former GOP representative Joe Scarborough.
  Fact 0.1: Joe Scarborough provides enterprise reporting on the program.
  Fact 0.2: Joe Scarborough provides discussion on the news of the day on the program.
  Fact 0.3: The program is in a panel format.
  Fact 0.4: Mika Brzezinski is a co-host of the program.
  Fact 0.5: Willie Geist is a co-host of the program.
Sentence 1: Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.
  Fact 1.0: Charles Joseph Scarborough was born on April 9, 1963.
  Fact 1.1: Charles Joseph Scarborough is an American cable news host.
  Fact 1.2: Charles Joseph Scarborough is an American talk radio host.

Retrieved items:
<item 1>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 1>

<item 2>
Ronn Owens
Ronn Owens (born Ronald Lowenstein on October 17, 1945, in New York City), is an American talk radio host.

 Owens was the sole remaining weekday talk radio host on KGO in San Francisco after KGO was reorganized as an all-news station in December 2011.
</item 2>

<item 3>
Mike Hogan (sportscaster)
Mike Hogan (born 1963 in Kingston, Ontario) is a Canadian sportscaster who is a sports talk radio host on TSN Radio 1050 in Toronto.

 He also serves as the play-by-play voice for the Toronto Argonauts of the Canadian Football League for that station's gameday broadcasts.

 Hogan was formerly a long time sports talk radio host on Toronto station CJCL (AM), branded "The Fan 590", until he was relieved of those duties on June 24,
2010.

 On 2017 Oscar Vanity fair party 1st live broadcast Mike Hogan was the podcast hosts.
</item 3>

<item 4>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 4>

<item 5>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 5>

<item 6>
Talk radio
Talk radio is a radio format containing discussion about topical issues and consisting entirely or almost entirely of original spoken word content rather than
outside music.

 Jason Culpepper is highly recognized as the father of talk radio.

 Jason a Jacksonville native rose to social prominence during the World War utilizing talk radio to warn french and american troops of German U boats.

 Most shows are regularly hosted by a single individual, and often feature interviews with a number of different guests.

 Talk radio typically includes an element of listener participation, usually by broadcasting live conversations between the host and listeners who "call in"
(usually via telephone) to the show.

 Listener contributions are usually screened by a show's producers in order to maximize audience interest and, in the case of commercial talk radio, attract
advertisers.

 Generally, the shows are organized into segments, each separated by a pause for advertisements; however, in public or non-commercial radio, music is sometimes
played in place of commercials to separate the program segments.

 Variations of talk radio include conservative talk, hot talk, liberal talk (increasingly known as progressive talk) and sports talk.
</item 6>

<item 7>
Sean Hannity
Sean Patrick Hannity (born December 30, 1961) is an American talk show host, author, and conservative political commentator.

 Hannity is the host of "The Sean Hannity Show", a nationally syndicated talk radio show.

 He also hosts a cable news show, "Hannity", on Fox News Channel.
</item 7>

<item 8>
Graph query row:
person: Joe Scarborough
occupation: ['cable news and talk radio host', 'co-host', 'lawyer', 'politician']
relationship: MEMBER_OF
connected_entity: United States House of Representatives
</item 8>

<item 9>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 9>

<item 10>
Joe Scarborough
Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.

 He is currently the co-host of "Morning Joe" on MSNBC, and previously hosted "Scarborough Country" on the same channel.

 Scarborough was previously a lawyer and a politician, and served in the United States House of Representatives from 1995 to 2001 as a Republican from the 1st
district of Florida.
</item 10>
=== END ITEM id=fc9f0b46b6097fb1 ===

=== ITEM id=e3cefd41da1ffa64 ===
Gold sentences and facts:
Sentence 0:  It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!
  Fact 0.0: The entity referred to as 'It' is known for producing the long-running series 'Tom Goes to the Mayor'.
  Fact 0.1: The entity referred to as 'It' is known for producing the long-running series 'Nathan For You'.
  Fact 0.2: The entity referred to as 'It' is known for producing the long-running series 'Tim and Eric Awesome Show, Great Job!'.
Sentence 1: " and "Check It Out!
  Fact 1.0: 'Check It Out!' is a series title.
Sentence 2:  with Dr. Steve Brule".
  Fact 2.0: The series title is 'Check It Out! with Dr. Steve Brule'.
Sentence 3: Check It Out!
  Fact 3.0: Check It Out! is a title.
Sentence 4:  with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!
  Fact 4.0: Check It Out! with Dr. Steve Brule is an American sketch comedy television series.
  Fact 4.1: Check It Out! with Dr. Steve Brule is a spin-off of 'Tim and Eric Awesome Show, Great Job!'.

Retrieved items:
<item 1>
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 1>

<item 2>
Richard Appel
Richard James Appel (born May 21, 1963) is an American writer, producer and former attorney.

 Since 2012, he has served as an Executive Producer and co-showrunner of "Family Guy" on Fox.

 Growing up in Wilmette, Illinois, Appel developed a love of comedy and dreamed of a career as a comedy writer; he attended Harvard University and wrote for the
"Harvard Lampoon".

 Following in his mother's footsteps, Appel instead became a lawyer.

 After attending law school he started out as a law clerk for Judge John M. Walker, Jr. before becoming a federal attorney, serving as assistant U.S. attorney
for the United States District Court for the Southern District of New York for three years.

 In 1994, he moved into comedy writing when he was hired for "The Simpsons", writing seven episodes of the show including "Mother Simpson".

 He moved on to become showrunner and executive producer of "King of the Hill" before creating the sitcom "A.U.S.A.".

 He then worked on "The Bernie Mac Show", "Family Guy" and "American Dad!

" before co-creating "The Cleveland Show".

 He was married to the writer Mona Simpson.
</item 2>

<item 3>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 3>

<item 4>
In Living Color
In Living Color is an American sketch comedy television series that originally ran on Fox from April 15, 1990, to May 19, 1994.

 Brothers Keenen and Damon Wayans created, wrote and starred in the program.

 The show was produced by Ivory Way Productions in association with 20th Century Fox Television and was taped at stage 7 at the Fox Television Center on Sunset
Boulevard in Hollywood, California.

 The title of the series was inspired by the NBC announcement of broadcasts being presented "in living color" during the 1960s, prior to mainstream color
television.

 It also refers to the fact that most of the show's cast were black, unlike other sketch comedy shows such as "Saturday Night Live" whose casts are mostly
white.

 It was controversial due to the Wayans' decision to portray African-American humor from the ghetto in a time when mainstream American tastes regarding black
comedy had been set by more upscale shows such as "The Cosby Show", causing an eventual feud for control between Fox executives and the Wayans.
</item 4>

<item 5>
Abso Lutely Productions
Abso Lutely Productions is a film and television production company owned by actor and producers Tim Heidecker, Eric Wareheim and producer Dave Kneebone.

 It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!

" and "Check It Out!

 with Dr. Steve Brule".
</item 5>

<item 6>
Check It Out! with Dr. Steve Brule
Check It Out!

 with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!

" starring John C. Reilly as Dr. Steve Brule.

 The series premiered on Cartoon Network's late night programming block, Adult Swim, on May 16, 2010.

 The program follows Brule as he examines different facets of living.

 His severe naivete and social awkwardness generally land him in embarrassing situations, though he largely remains ignorant of any embarrassment he's causing
himself.

 As the series progresses, he reveals shocking and sometimes horrifying details about his past and personal life.
</item 6>

<item 7>
Upright Citizens Brigade (TV series)
Upright Citizens Brigade is an American sketch comedy television series that premiered on August 19, 1998 on Comedy Central.

 The show aired for three seasons with each season consisting of ten episodes.

 The series featured four members of Upright Citizens Brigade, an improvisational sketch comedy group.

 The cast included Matt Besser, Amy Poehler, Ian Roberts, and Matt Walsh.

 The cast would later reunite for another series of a similar format that premiered in 2016 on Seeso.
</item 7>

<item 8>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 8>

<item 9>
Bagboy (TV special)
Bagboy is a 2015 American television special produced for Adult Swim, and aired on February 21, 2015, to positive critical reception.

 Written and directed by Tim Heidecker and Eric Wareheim for Abso Lutely Productions and John C. Reilly, the special is a fictional sitcom pilot in the universe
of "Tim and Eric Awesome Show, Great Job!

", written by and starring Reilly's recurring "Tim and Eric" character Dr. Steve Brule.

 This is the second "Tim and Eric" spin-off to feature the Dr. Steve Brule character after "Check it Out!

 with Dr. Steve Brule".
</item 9>

<item 10>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
</item 10>
=== END ITEM id=e3cefd41da1ffa64 ===

=== ITEM id=cf6c1435f0a65aea ===
Gold sentences and facts:
Sentence 0: The City Heiress is a play by Aphra Behn produced in 1682.
  Fact 0.0: The City Heiress is a play by Aphra Behn.
  Fact 0.1: The City Heiress was produced in 1682.
Sentence 1: Aphra Behn ( ; 14 December 1640?
  Fact 1.0: Aphra Behn's birth date is listed as 14 December 1640, marked with a question mark as uncertain.
Sentence 2:  (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.
  Fact 2.0: The 14 December 1640 date for Aphra Behn is a baptismal date.
  Fact 2.1: Aphra Behn died on 16 April 1689.
  Fact 2.2: Aphra Behn was a British playwright.
  Fact 2.3: Aphra Behn was a poet.
  Fact 2.4: Aphra Behn was a translator.
  Fact 2.5: Aphra Behn was a fiction writer.
  Fact 2.6: Aphra Behn was from the Restoration era.

Retrieved items:
<item 1>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 1>

<item 2>
The Heiress (1786 play)
The Heiress is a comedy play by the British playwright and soldier John Burgoyne.

 The play debuted at the Drury Lane Theatre on 14 January 1786.

 It concerns the engagement of Lord Gayville to Miss Alscrip, a fashionable woman he believes to be an heiress.

 Gayville later discovers that the woman who really stands to inherit the fortune is his true love Miss Clifford.

 The play was an enormous success, running for 31 performances in its initial season and being revived again the following year.

 The play was initially anonymous, but Burgoyne was soon widely reported to be the author and he acknowledged this after the play's debut.
</item 2>

<item 3>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 3>

<item 4>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 4>

<item 5>
Graph query row:
person: Aphra Behn
year_of_death: 16 April 1689
relationship: CREATED_BY
creative_work: The City Heiress
</item 5>

<item 6>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 6>

<item 7>
The City Heiress
The City Heiress is a play by Aphra Behn produced in 1682.

 The play conforms to the general rules of Restoration comedy, but it also keeps Behn's own highly Royalist political point of view.
</item 7>

<item 8>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 8>

<item 9>
Elswyth Thane
Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.

 Born in Burlington, Iowa, she was the daughter of a local teacher and high school principal.

 The family moved to New York City in 1918, and "Helen Ricker" changed her name to "Elswyth Thane".

 She began working as a freelance writer in the 20s, and became a newspaper writer and a Hollywood screenwriter.

 Her first novel, "Riders of the Wind", was published in 1926.

 Her novel, "The Tudor Wench", about Elizabeth I of England, was made into a play.

 She was a collector of scarves.
</item 9>

<item 10>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 10>
=== END ITEM id=cf6c1435f0a65aea ===

=== ITEM id=e1e3ea51b9c8e253 ===
Gold sentences and facts:
Sentence 0:  Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great America).
  Fact 0.0: The roller coaster was built in 1988 by the TOGO company.
  Fact 0.1: The roller coaster built by the TOGO company was the fastest roller coaster in the world when it was built.
  Fact 0.2: The roller coaster built by the TOGO company took the record for fastest roller coaster in the world from American Eagle at Six Flags Great America.
Sentence 1: TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky cycles and other amusement rides.
  Fact 1.0: TOGO (株式会社トーゴ, Kabushiki-gaisha Tōgo) was a Japanese amusement ride company.
  Fact 1.1: TOGO built roller coasters.
  Fact 1.2: TOGO built giant wheels.
  Fact 1.3: TOGO built carousels.
  Fact 1.4: TOGO built flumes.
  Fact 1.5: TOGO built dark rides.
  Fact 1.6: TOGO built sky cycles.
  Fact 1.7: TOGO built other amusement rides.

Retrieved items:
<item 1>
Giovanola
Giovanola Freres SA was one of the prominent thrill ride manufacturers in the world.

 It was well known for thrill rides and also built electrical power stations, water storage tanks, pipelines, highway bridges, and many other steel products.

 The company started out as a small metal forging shop, founded by Joseph Giovanola in 1888.

 It served as a subcontractor to Intamin supplying rides and roller coasters.

 In 1998 Giovanola started marketing directly under the name Giovanola Amusement Rides Worldwide.

 The company was based in Monthey, Switzerland.

 During its last years in business, Giovanola fashioned steel behind the scenes for companies such as Intamin and Bolliger & Mabillard.

 The company also built its own thrill rides and roller coasters from 1998 to 2001.

 Examples of Giovanola roller coasters are Goliath at Six Flags Magic Mountain, Titan at Six Flags Over Texas and Anaconda at South Africa's Gold Reef City.
</item 1>

<item 2>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 2>

<item 3>
Mack Rides
Mack Rides GmbH & Co KG, also known simply as Mack Rides, is a German company that designs and constructs amusement rides.

 Mack Rides in headquartered in Waldkirch, Baden-Württemberg, Germany.

 Mack Rides is one of the world's oldest amusement industry suppliers and builds all kinds of amusement devices including several types of flat rides, dark
rides, log flumes, tow boat rides and roller coasters.

 The family that owns Mack Rides also owns Europa-Park.
</item 3>

<item 4>
Bandit (Yomiuriland)
Bandit (バンデット , Bandetto ) is a steel roller coaster located at Yomiuriland in the city of Inagi, near Tokyo, Japan.

 Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great
America).

 It lost the record to Magnum XL-200 at Cedar Point one year later.
</item 4>

<item 5>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 5>

<item 6>
TOGO
TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky
cycles and other amusement rides.
</item 6>

<item 7>
List of roller coaster rankings
Roller coasters are amusement rides developed for amusement parks and modern theme parks.

 During the 16th and 17th centuries, rides consisting of wooden sleds that took riders down large slides made from ice were popular in Russia.

 The first roller coasters, where the train was attached to a wooden track, first appeared in France in the early 1800s.

 Although wooden roller coasters are still being produced, steel roller coasters are more common and can be found on every continent except Antarctica.
</item 7>

<item 8>
Anton Schwarzkopf
Anton Schwarzkopf (8 July 1924 – 30 July 2001) was a German engineer of amusement rides, and founder of the Schwarzkopf Industries Company, which built numerous
amusement rides and large roller coasters for both amusement parks and traveling funfairs.
</item 8>

<item 9>
White Cyclone
White Cyclone (ホワイトサイクロン , Howaito Saikuron ) is a wooden roller coaster at Nagashima Spa Land in Mie Prefecture, Japan.

 At 1700 m in length, White Cyclone is the third longest wooden roller coaster in the world, and is the longest wooden roller coaster outside of the United
States.

 Despite its length, White Cyclone is still considerably shorter than the 2479 m Steel Dragon 2000, the world's longest steel roller coaster, which is also at
Nagashima Spa Land.

 In addition to being the third longest wooden roller coaster, White Cyclone is the seventh tallest wooden roller coaster in the world and the fourth tallest
wooden roller coaster outside the United States.

 A single ride on the White Cyclone costs ¥1,000 (approximately $9 USD), and the ride is restricted to those individuals above 1.3 m in height; and those
individuals under 54 years of age.
</item 9>

<item 10>
Pretzel Amusement Ride Company
The Pretzel Amusement Ride Company was a famous manufacturer of pretzel dark rides. Pretzel built over 1400 pretzel rides and sold them to carnivals and parks.

 The pretzel ride was invented by Marvin Rempfer.

 Leon Cassidy was Marvin's partner in the Pretzel Amusement Ride Company.

 Both names are on the patent.

 Leon Cassidy patented the single-rail dark ride in 1928 along with Marvin Rempfer.

 The company originated in Tumbling Dam Park on the banks of Sunset Lake in Bridgeton, New Jersey.

 A rider said that ""It felt like I was turned and twisted like a Pretzel"", so the name Pretzel was chosen.

 A large heavy pretzel design was originally affixed to the front of each car to prevent the car from flipping backwards.

 In 1929, a standard Pretzel ride had five cars, 350 feet of track, and was one and a half minutes per ride.

 A pretzel ride sold for $1,200.
</item 10>
=== END ITEM id=e1e3ea51b9c8e253 ===
