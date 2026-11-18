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

This file has 11 items.

=== ITEM id=cdae6a5da0778bcb ===
Gold sentences and facts:
Sentence 0:  After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.
  Fact 0.0: He spent some time in general practice.
  Fact 0.1: He became a Lecturer in animal behaviour at De Montfort University.
  Fact 0.2: He became a Senior Lecturer in animal behaviour at De Montfort University.
  Fact 0.3: He became a Principal Lecturer in animal behaviour at De Montfort University.
  Fact 0.4: He subsequently moved to the University of Lincoln.
  Fact 0.5: He has been Head of the Department of Biological Sciences at the University of Lincoln.
Sentence 1:  Individuals who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title of "Doctor" (often abbreviated "Dr") or, in non-English speaking countries, variants such as "Dr. phil."
  Fact 1.0: People who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title 'Doctor'.
  Fact 1.1: The title 'Doctor' is often abbreviated 'Dr'.
  Fact 1.2: In non-English speaking countries, people with a Doctor of Philosophy degree may use variants such as 'Dr. phil.'

Retrieved items:
<item 1>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 1>

<item 2>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 2>

<item 3>
Steven K. Galson
Steven Kenneth Galson (born 1956) is an American public health physician.

 He is currently Senior Vice President for Global Regulatory Affairs at Amgen, the S. California-based innovative global biopharmaceutical company.

 He is also Professor-at-Large at the Keck Graduate Institute for Applied Life Sciences in Clarmont, California.

 He is a retired rear admiral in the United States Public Health Service Commissioned Corps and public health administrator who served as the acting Surgeon
General of the United States from October 1, 2007 – October 1, 2009.

 He served concurrently as acting Assistant Secretary for Health from January 22, 2009 to June 25, 2009, and as the Deputy Director and Director of the Center
for Drug Evaluation and Research (CDER) at the Food and Drug Administration from 2001 to 2007.

 As the Acting Surgeon General, he was the commander of the United States Public Health Service Commissioned Corps and, while serving as the Assistant Secretary
for Health, was the operational head of the Public Health Service.
</item 3>

<item 4>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 4>

<item 5>
Antje Boetius
Antje Boetius (born 5 March 1967) is a German marine biologist presently serving as professor of geomicrobiology at the Max Planck Institute for Marine
Microbiology, University of Bremen.

 She received the Gottfried Wilhelm Leibniz Prize, with 2.5 million euros in funding, in March 2009 for her study of sea bed microorganisms that affect the
global climate.

 She was the first person to describe anaerobic oxidation of methane, and believes the Earth's earliest life forms may have subsisted on methane in the absence
of molecular oxygen (instead reducing oxygen-containing compounds such as nitrate or sulfate).

 She has also suggested such life forms may be able to reduce the rate of climate change in future.
</item 5>

<item 6>
Daniel Mills (biologist)
Daniel Simon Mills, FRCVS (born 21 Aug 1966) is an English biologist and the UK's first Professor of Veterinary Behavioural Medicine based at the University of
Lincoln, United Kingdom.

 He attended St Edmund's College, Ware before studying at the University of Bristol where he received his BVSc degree and completed his Ph.D. degree in animal
behaviour from De Montfort University.

 After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and
subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.

 His research has focused on the assessment of emotion in companion animals and the use of semiochemicals to manage their problem behaviour (pheromonatherapy).
</item 6>

<item 7>
David M. Sever
David M. Sever is an American herpetologist, histologist, anatomist and reproductive biologist.

 He has been a professor and department head in the Department of Biological Sciences at Southeastern Louisiana University since 2004, and held the Kenneth
Dyson Endowed Professorship in Biological Sciences from 2012 to 2015.

 He is well known for over 30 years of research on the secondary sexual characteristics of salamanders and more generally on comparative histoanatomy of the
urogenital systems of vertebrates.

 and was recognized as the 2013 Distinguished Herpetologist of the Year by the Herpetologists' League.
</item 7>

<item 8>
Richard F. Wilson
Richard F. Wilson was selected to be Illinois Wesleyan University's 18th President on April 6, 2004 and took office on July 1, 2004.

 He was inaugurated on April 9, 2005.

 He had spent the previous 26 years at the University of Illinois at Urbana-Champaign, leaving as associate chancellor for development and vice president of the
University of Illinois Foundation.

 Mr. Wilson is a Ph.D. graduate of the University of Michigan.

 Wilson announced his intended retirement from Illinois Wesleyan, to take place around June 2015 or whenever a successor is named and transitioned.
</item 8>

<item 9>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 9>

<item 10>
William McGinnis
William "Bill" McGinnis, Ph.D. is a molecular biologist and professor of biology at the University of California San Diego.

 At UC San Diego he has also served as the Chairman of the Department of Biology from July 1998 - June 1999, as Associate Dean of the Division of Natural
Sciences from July 1, 1999 - June 2000, and as Interim Dean of the newly established Division of Biological Sciences from July 1, 2000 - February 1, 2001.

 Dr. McGinnis was appointed Dean of the Divisional Biological Sciences on July 1, 2013
</item 10>
=== END ITEM id=cdae6a5da0778bcb ===

=== ITEM id=5fc2830f6453ebb9 ===
Gold sentences and facts:
Sentence 0:  He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.
  Fact 0.0: He is mainly known for his spells at Real Sociedad.
  Fact 0.1: His offensive partnership with Nihat Kahveci at Real Sociedad was one of the best in Spain.
Sentence 1: Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
  Fact 1.0: Nihat Kahveci was born on 23 November 1979.
  Fact 1.1: Nihat Kahveci is a Turkish former footballer.
  Fact 1.2: Nihat Kahveci played as a forward.

Retrieved items:
<item 1>
Real Sociedad Femenino
Real Sociedad Femenino is the women's football section of Real Sociedad de Fútbol.

 Founded in 2004
</item 1>

<item 2>
Ivan Toplak
Ivan Toplak (Serbian Cyrillic: Ивaн Toплaк; born September 21, 1931 in Belgrade) is a former Yugoslavian football player and manager.

 He earned 1 cap for Yugoslavia.

 He started his career at Branik in Maribor, where his family moved in 1943 from Belgrade.
</item 2>

<item 3>
José Antonio Santamaría Mikel Vaqueriza
José Antonio Santamaría Mikel Vaqueriza (born 16 March 1946 in San Sebastián, Guipúzcoa, Spain; died 19 January 1993) was a Spanish footballer.

 During his career as a professional footballer, Santamaría played for SD Eibar (1963 to 1964), Real Sociedad (1964 to 1967), Real Sociedad (1967 to 1971),
Hercules CF (1971 to 1974) finishing his career at CE Sabadell FC (1974 to 1975).

 After retiring from football he had a career as an entrepreneur in the hospitality sector.

 He was killed in an attack committed by ETA in 1993.
</item 3>

<item 4>
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 4>

<item 5>
Darko Kovačević
Darko Kovačević (] ; Serbian Cyrillic: ; born 18 November 1973) is a Serbian former footballer who played as a forward.

 He began his career in Serbia with Proleter Zrenjanin and subsequently played for Red Star Belgrade, with whom he won a Yugoslav League title and two Yugoslav
Cups.

 His prolific performances earned him a move to Premier League side Sheffield Wednesday, although his time in England was less successful.

 He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.

 Kovačević also had positive spells with Italian club Juventus and Greek side Olympiacos.

 At international level, he represented Yugoslavia at the 1998 FIFA World Cup and at UEFA Euro 2000.
</item 5>

<item 6>
Real Sociedad
Real Sociedad de Fútbol, S.A.D., more commonly referred to as Real Sociedad (] ; "Royal Society") or La Real, is a Spanish football club based in the city of
San Sebastián, Basque Country, founded on 7 September 1909.

 It plays its home matches at the 32,000-capacity Anoeta Stadium.

 Real Sociedad won the Liga title in 1980–81 and 1981–82, and last finished runners-up in 2002–03.

 The club has also won the Copa del Rey twice, in 1909 and 1987.

 It contests the Basque derby against rivals Athletic Bilbao.

 Real Sociedad were founder members of La Liga in 1928, and its longest spell in the top flight was for 40 seasons, from 1967 to 2007.
</item 6>

<item 7>
2013–14 Real Sociedad season
The 2013–14 season was Real Sociedad's 67th season in La Liga.

 Real Sociedad finished 7th in the league and reached the semifinals of the Copa del Rey.

 The Basque failed to make it out of the group stages of the UEFA Champions League.
</item 7>

<item 8>
Sport Mundi Tournament
The Sport Mundi Tournament is a Spanish pre-season women's football invitational charitable tournament held every August in Irun since 2005.

 It is contested by four teams, usually including regional powerhouses Athletic Bilbao and Real Sociedad.

 Athletic is the most successful team in the competition with four titles, followed by Levante UD and Real Sociedad with two each.
</item 8>

<item 9>
C.D. Real Sociedad
Club Deportivo Real Sociedad, commonly known as Real Sociedad (] ), is a Honduran football club based on Tocoa, Colón, Honduras.
</item 9>

<item 10>
Nihat Kahveci
Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
</item 10>
=== END ITEM id=5fc2830f6453ebb9 ===

=== ITEM id=6eb5941c8fdbb57a ===
Gold sentences and facts:
Sentence 0:  It featured Reese Witherspoon and Paul Rudd, prior to both becoming considerably bigger film stars.
  Fact 0.0: The film featured Reese Witherspoon and Paul Rudd.
  Fact 0.1: Reese Witherspoon and Paul Rudd were in the film before both became considerably bigger film stars.
Sentence 1:  Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for which she was nominated for a Young Artist Award.
  Fact 1.0: The actress was born in New Orleans.
  Fact 1.1: The actress was raised in Tennessee.
  Fact 1.2: The actress began her career as a child actress.
  Fact 1.3: The actress made her professional screen debut in 'The Man in the Moon' (1991).
  Fact 1.4: The actress was nominated for a Young Artist Award for her debut in 'The Man in the Moon'.

Retrieved items:
<item 1>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 1>

<item 2>
Doreen Ketchens
Doreen Ketchens (born October 3, 1966 in New Orleans, Louisiana) is a jazz clarinetist, who performs Dixieland and Trad Jazz.

 She is one of the first and few female bandleaders in New Orleans, and a musical educator.

 She has performed at concert halls and music festivals, at U.S. Embassies and decades of weekly performances in Dixieland's tradition in the Royal Street
Performing Arts Zone in the French Quarter of New Orleans with her band, Doreen's Jazz New Orleans.

 Ketchens has performed for four U.S. Presidents: Bill Clinton, George Bush Sr., Ronald Reagan and Jimmy Carter, and is widely considered one of the cultural
ambassadors of New Orleans and of the traditional music
</item 2>

<item 3>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 3>

<item 4>
City of New Orleans (train)
The City of New Orleans is an Amtrak passenger train which operates on an overnight schedule between Chicago and New Orleans.

 The train is a successor to the Illinois Central Railroad's "Panama Limited".

 The present name was revived in 1981.
</item 4>

<item 5>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 5>

<item 6>
Paul Soniat
Paul Soniat is the Director of the New Orleans City Park Botanical Gardens.

 He was born at the old Touro Hospital in New Orleans.

 His family has been in New Orleans since 1727 and he grew up and later lived on the street bearing his family name.

 Paul is a self-taught piano player, and has released two CDs, absorbing the sights, sounds, and flavor of New Orleans.

 His first CD, titled "Born in New Orleans", was released in April 2005.

 His second CD was released after Hurricane Katrina hit the Gulf coast and the New Orleans' levees failed.

 This second CD is appropriately titled "Below the Water Line".
</item 6>

<item 7>
Overnight Delivery
Overnight Delivery is a 1998 romantic comedy film directed by Jason Bloom.

 It was rated PG-13 by the MPAA and released direct-to-video.

 It featured Reese Witherspoon and Paul Rudd, prior to both becoming considerably bigger film stars.
</item 7>

<item 8>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 8>

<item 9>
Reese Witherspoon
Laura Jeanne Reese Witherspoon ( ; born March 22, 1976) is an American actress, producer, and entrepreneur.

 Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for
which she was nominated for a Young Artist Award.

 Following breakout roles in "" (1992) and "Jack the Bear" (1993), she starred in the comedy-drama "Pleasantville" (1998), for which she won the Young Hollywood
Award for Breakthrough Performance.

 Her leading role of Tracy Flick in "Election" (1999) was nominated for a Golden Globe Award.
</item 9>

<item 10>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 10>
=== END ITEM id=6eb5941c8fdbb57a ===

=== ITEM id=d4066a3d430b657f ===
Gold sentences and facts:
Sentence 0: The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
  Fact 0.0: The Albanian Coalition 'Perspective' is an Albanian political party.
  Fact 0.1: The Albanian Coalition 'Perspective' is in Montenegro.
Sentence 1: Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.
  Fact 1.0: Albanians in Montenegro constitute 4.91% of the county's total population.
  Fact 1.1: The Albanian name for Albanians in Montenegro is 'Shqiptarët e Malit të Zi'.

Retrieved items:
<item 1>
Orlov let
Orlov let ("Eagle's Flight") was the name of an anti-terrorist operation conducted by the Montenegrin police to arrest a group of Albanians who planned
terrorist attacks and an armed conflict in Albanian-inhabited parts of Montenegro.

 The group of 17 people planned operations from 2004 to their arrest on the night of 10/11 September 2006.

 Weapons and explosive material were found in police raids.

 The group's main operative was the destroying of cultural and religious buildings (Orthodox) in Albanian-inhabited territory.

 The group was organised by an Albanian association based in Detroit, and members of the Kosovo Liberation Army, from which they received financial aid and
smuggled arms into Montenegro.

 The group was given a total sentence of 51 years of prison.

 Relations between the state and the Albanian minority strained after the event.
</item 1>

<item 2>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 2>

<item 3>
Albanians in Montenegro
Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.

 Albanians of Montenegro are Ghegs who mainly live in southeastern and eastern Montenegro, mainly in the following municipalities: Ulcinj (71% of population),
Plav (19%), Bar (6%), Podgorica (5%) and Rožaje (5%).
</item 3>

<item 4>
Albanians in the Republic of Macedonia
Albanians are the largest ethnic minority in the Republic of Macedonia.

 Of the 2,022,547 citizens of Macedonia, 509,083, or 25.2%, are Albanian according to the latest national census in 2002.

 The Albanian minority lives mostly in the north-western part of the country.

 The largest Albanian communities are in the municipalities of Tetovo (70.3% of the total population), Gostivar (66.7%), Debar (58.1%), Struga (56.8%), Kičevo
(54.5%), Kumanovo (25.8%) and Skopje (20.4%).
</item 4>

<item 5>
Albania
Albania is a unitary parliamentary constitutional republic with the capital in Tirana, the country's largest city and main economic and commercial centre,
followed by Durrës.

 The country's other major cities include Vlorë, Sarandë, Shkodër, Berat, Korçë, Gjirokastër and Fier.
</item 5>

<item 6>
Albania
Albania ( , ; Albanian: "Shqipëri/Shqipëria" ; Gheg Albanian: "Shqipni/Shqipnia or Shqypni/Shqypnia" ), officially the Republic of Albania (Albanian: "Republika
e Shqipërisë" , ] ), is a country in Southern and Southeastern Europe.

 The country spans 28,748 km2 and had a total population of almost 3 million people as of 2016 .

 Albania is located in the southwestern part of the Balkan Peninsula, bordered by Montenegro to the northwest, Kosovo to the northeast, the Republic of
Macedonia to the east, and Greece to the south and southeast.

 The country has a coastline on the northern shore of the Mediterranean Sea, the Adriatic Sea to the west and the Ionian Sea to the southwest, forming the
Albanian Riviera.

 Albania is less than 72 km from Italy, across the Strait of Otranto which connects the Adriatic Sea to the Ionian Sea.
</item 6>

<item 7>
Minority languages of Montenegro
European Charter for Regional or Minority Languages entered into force in Montenegro in June 2006, following the independence of Montenegro from the State Union
of Serbia and Montenegro on 3 June 2006.

 Constitution of Montenegro from 2007 states that Montenegrin language is official language of country, while Bosnian, Croatian, Serbian and Albanian are
languages in official use.

 The Constitution states that languages in official use are those of groups that form at least 1% of the population of Montenegro, as per 2003 population
census.

 The Law on National Minorities specifies that the percentage of members of national minorities in total population of the local government should be 15% in
order for their language and script to be introduced in official use.

 Media whose founder is Montenegro government are obliged to broadcast news, cultural, educational, sports and entertainment programs in minority languages.

 Minorities and their members have the right to education in their language in regular and vocational education.
</item 7>

<item 8>
Darza
Darza (Albanian: "Darzë" , Montenegrin and ) is a village in the Ulcinj Municipality, southeastern Montenegro.

 It is a multi-ethnic settlement, inhabited by Montenegrins, Serbs and Albanians.

 According to the 2003 census, the total population was 119.
</item 8>

<item 9>
Albanian Coalition &quot;Perspective&quot;
The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
</item 9>

<item 10>
Albanians
Albanians (Albanian: "Shqiptarët" ) are a nation and ethnic group, native to Albania, Kosovo and other countries who share a common culture, ancestry and speak
the Albanian language as a native tongue.

 Legally, the term is used to refer to the citizens of the Republic of Albania.

 Ethnic Albanians speak the Albanian language and more than half of ethnic Albanians live in Albania and Kosovo.

 A large Albanian population lives in Greece, Italy, the Republic of Macedonia, with smaller Albanian populations located in Serbia and Montenegro.
</item 10>
=== END ITEM id=d4066a3d430b657f ===

=== ITEM id=f01dfee78d23aae4 ===
Gold sentences and facts:
Sentence 0:  In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site LiveJasmin.
  Fact 0.0: In the category of pornographic websites, this site was surpassed in the rankings by xHamster.
  Fact 0.1: In the category of pornographic websites, this site was surpassed in the rankings by XVideos.
  Fact 0.2: In the category of pornographic websites, this site was surpassed in the rankings by Pornhub.
  Fact 0.3: In the category of pornographic websites, this site was surpassed in the rankings by the adult webcam site LiveJasmin.
  Fact 0.4: xHamster, XVideos, and Pornhub are competitor sites.
Sentence 1: XVideos is the world's largest pornographic video sharing and viewing website.
  Fact 1.0: XVideos is the world's largest pornographic video sharing and viewing website.

Retrieved items:
<item 1>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 1>

<item 2>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 2>

<item 3>
Pravda.ru
Pravda.ru (formerly Pravda Online) is a Russian internet news website established in 1999 and owned by Pravda.ru Holding (headed by Vadim Gorshenin).
</item 3>

<item 4>
YouPorn
YouPorn is a free pornographic video sharing website and one of the 100 most accessed websites in the world.

 Since launching in August 2006, it grew to become the most popular pornographic website on the internet, and, in November 2007, it was reported to be the
largest free pornographic website as well.

 As of February 2013, it was the 83rd most popular website overall and the fifth most popular pornographic website.

 In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site
LiveJasmin.
</item 4>

<item 5>
XY TV
It launched in December 2003 and distribution was limited to select cable companies.

 As of May 31, 2006, XY.tv has ceased operations, and is no longer broadcasting.

 It was wholly owned by American radio personality John Garabedian.
</item 5>

<item 6>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 6>

<item 7>
Pioneer Pekao Investments
Pioneer Pekao is a Polish company managing several investment funds in Poland.

 It is the biggest and first investment funds manager on the Polish market.
</item 7>

<item 8>
XHamster
xHamster is a pornographic media and social networking site headquartered in Limassol, Cyprus.

 xHamster serves user-submitted pornographic videos, webcam models, pornographic photographs, and erotic literature, and incorporates social networking
features.

 xHamster was founded in 2007.

 With more than 10 million members, it is the third most popular pornography website on the Internet after XVideos and Pornhub.
</item 8>

<item 9>
XVideos
XVideos is the world's largest pornographic video sharing and viewing website.

 It's registered to the Polish company, WGCZ Holding.

 Alexa ranks it as the 50th most visited website in the world.
</item 9>

<item 10>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 10>
=== END ITEM id=f01dfee78d23aae4 ===

=== ITEM id=333a6c7a18d87a9e ===
Gold sentences and facts:
Sentence 0: Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.
  Fact 0.0: Buried Country is a highly regarded documentary film
  Fact 0.1: Buried Country is a highly regarded book
  Fact 0.2: Buried Country is a highly regarded soundtrack album
  Fact 0.3: Buried Country is now also a stageshow
Sentence 1: South of the Border is a 2009 American documentary film directed by Oliver Stone.
  Fact 1.0: South of the Border is a documentary film
  Fact 1.1: South of the Border is American
  Fact 1.2: South of the Border is from 2009
  Fact 1.3: South of the Border was directed by Oliver Stone

Retrieved items:
<item 1>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 1>

<item 2>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 2>

<item 3>
Station to Station (film)
Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.

 It was distributed internationally in 2015 following its release at the Sundance Film Festival.
</item 3>

<item 4>
No eres tú, soy yo
No eres tú, soy yo (English: It's not you, it's me ) is a 2010 Mexican romantic comedy directed by Alejandro Springall starring Eugenio Derbez, Alejandra Barros
and Martina García and based on the Argentinean film "No sos vos, soy yo" (2004), written by Juan Taratuto.

 It was produced by Matthias Eherenberg and filmed in Mexico City.
</item 4>

<item 5>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 5>

<item 6>
The Colleen Bawn
The Colleen Bawn, or The Brides of Garryowen is a melodramatic play written by Irish playwright Dion Boucicault.

 It was first performed at Miss Laura Keene's Theatre, New York, on 27 March 1860 with Laura Keene playing Anne Chute and Boucicault playing Myles na Coppaleen.

 It was most recently performed in Dublin at the Project Arts Centre in July and August 2010.

 Several film versions have also been made.
</item 6>

<item 7>
The Country Girl (1954 film)
The Country Girl is a 1954 American drama film directed by George Seaton and starring Bing Crosby, Grace Kelly, and William Holden.

 Adapted by George Seaton from Clifford Odets' 1950 play of the same name, the film is about an alcoholic has-been actor struggling with the one last chance
he's been given to resurrect his career.

 Seaton won the Academy Award for Best Writing, Adapted Screenplay.

 It was entered in the 1955 Cannes Film Festival.
</item 7>

<item 8>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 8>

<item 9>
Buried Country
Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.

 It is a prosopography, created by noted writer Clinton Walker, that tells the story of Australian country music in the Aboriginal community by focussing on the
genre's most important stars.
</item 9>

<item 10>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 10>
=== END ITEM id=333a6c7a18d87a9e ===

=== ITEM id=f436fe219f9244d6 ===
Gold sentences and facts:
Sentence 0: Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
  Fact 0.0: Leitneria floridana is also called corkwood.
  Fact 0.1: Leitneria floridana is the sole species in the genus Leitneria.
  Fact 0.2: Leitneria floridana is a deciduous dioecious shrub or small tree.
  Fact 0.3: Leitneria floridana is found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
Sentence 1: Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.
  Fact 1.0: Tricyrtis is a genus of Asian flowering plants.
  Fact 1.1: Tricyrtis is in the lily family.
  Fact 1.2: Tricyrtis has approximately 20 known species.

Retrieved items:
<item 1>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 1>

<item 2>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 2>

<item 3>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 3>

<item 4>
Eucryphia: seven mostly evergreen South American/Australian trees; classified in Cunoniaceae. Bothriochloa ischaemum: widespread grass; var. songarica invasive
in Texas. Eriogonum buckwheats: genus diverse across North America; E. heracleoides inhabits western U.S.; E. diclinum forms low mats on serpentine soils; E.
crosbyae occurs in Oregon/Nevada.
</item 4>

<item 5>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 5>

<item 6>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 6>

<item 7>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 7>

<item 8>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 8>

<item 9>
Lepidozamia comprises two Australian rainforest cycads (2n=18). Orange Field, a Sri Lankan tea factory founded in 1950, makes about 500,000 kg of black tea
annually for Middle Eastern and CIS markets, sourcing from 750 smallholders. Lithocarpus has 334 accepted East/Southeast Asian species; three Java endemics
listed here are endangered, while L. glaber occurs in Japan, China and Taiwan. Oaks (Querc
</item 9>

<item 10>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 10>
=== END ITEM id=f436fe219f9244d6 ===

=== ITEM id=9a537134eb7e84df ===
Gold sentences and facts:
Sentence 0: William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.
  Fact 0.0: William Dwight Schultz was born on November 24, 1947.
  Fact 0.1: William Dwight Schultz is an American actor.
  Fact 0.2: William Dwight Schultz is an American voice artist.
Sentence 1:  He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the Squirrel in "CatDog".
  Fact 1.0: William Dwight Schultz is well known in animation as the mad scientist Dr. Animo in the "Ben 10" series.
  Fact 1.1: William Dwight Schultz is well known in animation as Chef Mung Daal in the children's cartoon "Chowder".
  Fact 1.2: William Dwight Schultz is well known in animation as Eddie the Squirrel in "CatDog".
Sentence 2: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 2.0: Chowder is an American animated television series.
  Fact 2.1: Chowder was created by C. H. Greenblatt.
  Fact 2.2: Chowder was created for Cartoon Network.

Retrieved items:
<item 1>
Rodger Bumpass
Rodger Bumpass (born November 20, 1951) is an American actor and voice actor.

 He is best known for his long-running role as Squidward Tentacles on the American animated television series "SpongeBob SquarePants".

 He voices many other characters on the show, including Dr. Forrest and various anchovies.

 He also voiced The Chief in the animated series "Where on Earth Is Carmen Sandiego?

", and Mr. Besser, the school principal in the animated series "The Kids from Room 402".

 Bumpass has many other credits in animated films, animated television series, and video games.
</item 1>

<item 2>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 2>

<item 3>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 3>

<item 4>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 4>

<item 5>
Dan Castellaneta
Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.

 Noted for his long-running role as Homer Simpson on the animated television series "The Simpsons", he also voices many other characters for the show, including
Abraham "Grampa" Simpson, Barney Gumble, Krusty the Clown, Sideshow Mel, Groundskeeper Willie, Mayor Quimby and Hans Moleman.
</item 5>

<item 6>
Sherry Alberoni
Sharyn Eileen "Sherry" Alberoni (born December 4, 1946) is an American actress and voice artist.

 Alberoni got her start as a Mouseketeer on the weekday ABC television program "The Mickey Mouse Club".

 As an adult, she became a voice artist for Hanna-Barbera Productions.

 Besides providing voices for numerous incidental characters in series such as "Jeannie", Alberoni is best known as the voice of nasty rich-girl Alexandra Cabot
from "Josie and the Pussycats" , "superhero-in-training" Wendy from the first season of "Super Friends", the heroic robot, Bo in "Mighty Orbots" and
Glumdalclitch in "The Three Worlds of Gulliver".

 In 1971, she starred alongside Patty Andrews in the Sherman Brothers stage musical, "Victory Canteen".
</item 6>

<item 7>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 7>

<item 8>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 8>

<item 9>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 9>

<item 10>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 10>
=== END ITEM id=9a537134eb7e84df ===

=== ITEM id=07b89948067eae01 ===
Gold sentences and facts:
Sentence 0:  The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova Planta Decree of Majorca and Ibiza (28 November 1715).
  Fact 0.0: The body that was to become the Gran i General Consell was founded in 1249.
  Fact 0.1: The Consell was abolished on 22 July 1718.
  Fact 0.2: The Consell was abolished by Philip V.
  Fact 0.3: Philip V was empowered by the Nova Planta Decree of Majorca and Ibiza.
  Fact 0.4: The Nova Planta Decree of Majorca and Ibiza is dated 28 November 1715.
Sentence 1: Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January 1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
  Fact 1.0: Philip V's Spanish name is Felipe V.
  Fact 1.1: Philip V's French name is Philippe.
  Fact 1.2: Philip V's Italian name is Filippo.
  Fact 1.3: Philip V was born on 19 December 1683.
  Fact 1.4: Philip V died on 9 July 1746.
  Fact 1.5: Philip V was King of Spain from 1 November 1700 to 15 January 1724.
  Fact 1.6: Philip V abdicated in favour of his son Louis on 15 January 1724.
  Fact 1.7: Philip V reassumed the Spanish throne on 6 September 1724, upon his son's death.
  Fact 1.8: Philip V was King of Spain from 6 September 1724 until his death on 9 July 1746.

Retrieved items:
<item 1>
Viscount Galway
Viscount Galway is a title that has been created once in the Peerage of England and thrice in the Peerage of Ireland.

 The first creation came in the Peerage of England in 1628 in favour of Richard Burke, 4th Earl of Clanricarde.

 He was made Earl of St Albans at the same time (see the Earl of Clanricarde for more information on this creation).

 The second creation came in the Peerage of Ireland in 1687 in favour of Ulick Bourke.

 He was made Baron Tyaquin at the same time, also in the Peerage of Ireland.

 However, both titles became extinct on his early death in 1691.

 The third creation came in the Peerage of Ireland in 1692 in favour of the French soldier and diplomat Henry de Massue, Marquis de Ruvigny.

 He was made Earl of Galway in 1697.

 However, both titles became extinct on his death in 1720.
</item 1>

<item 2>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 2>

<item 3>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 3>

<item 4>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 4>

<item 5>
House of Hanover
The House of Hanover (or the Hanoverians ; German: "Haus Hannover" ) is a German royal dynasty that ruled the Electorate and then the Kingdom of Hanover, and
that also provided monarchs of Great Britain and Ireland from 1714 and ruled the United Kingdom until the death of Queen Victoria in 1901.

 Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
</item 5>

<item 6>
Philip V of Spain
Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January
1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
</item 6>

<item 7>
Duke of Bedford
Duke of Bedford (named after Bedford, England) is a title that has been created six times (for five distinct people) in the Peerage of England.

 The first and second creations came in 1414 in favour of Henry IV's third son, John, who later served as regent of France.

 He was made Earl of Kendal at the same time and was made Earl of Richmond later the same year.

 The titles became extinct on his death in 1435.

 The third creation came in 1470 in favour of George Neville, nephew of Warwick the Kingmaker.

 He was deprived of the title by Act of Parliament in 1478.

 The fourth creation came 1478 in favour of George, the third son of Edward IV.

 He died the following year at the age of two.

 The fifth creation came in 1485 in favour of Jasper Tudor, half-brother of Henry VI and uncle of Henry VII.

 He had already been created Earl of Pembroke in 1452.

 However, as he was a Lancastrian, his title was forfeited between 1461 and 1485 during the predominance of the House of York.
</item 7>

<item 8>
Duke of Bedford
He regained the earldom in 1485 when his nephew Henry VII came to the throne and was elevated to the dukedom the same year.

 He had no legitimate children and the titles became extinct on his death in 1495.
</item 8>

<item 9>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 9>

<item 10>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 10>
=== END ITEM id=07b89948067eae01 ===

=== ITEM id=60959d9b3d718899 ===
Gold sentences and facts:
Sentence 0: Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.
  Fact 0.0: Vytautas Kevin Gerulaitis was born on July 26, 1954.
  Fact 0.1: Vytautas Kevin Gerulaitis died on September 17, 1994.
  Fact 0.2: Vytautas Kevin Gerulaitis was an American professional tennis player.
Sentence 1: Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.
  Fact 1.0: Heather Miriam Watson was born on 19 May 1992.
  Fact 1.1: Heather Miriam Watson is a British tennis player.
  Fact 1.2: Heather Miriam Watson is a Wimbledon Mixed Doubles Champion.

Retrieved items:
<item 1>
Ernests Gulbis
Ernests Gulbis (] , born 30 August 1988, and nicknamed "The Gull" or "Ernie") is a Latvian professional tennis player.

 In 2008, Gulbis won his first ATP Tour doubles title at the U.S. Men's Clay Court Championships, teaming with Rainer Schüttler, and in 2010 won his first ATP
Tour singles title in the Delray Beach, defeating Ivo Karlović in the final.

 In total, Gulbis has 6 ATP titles to his name and has never lost a final.

 His best performance at a Grand Slam is reaching the semifinals of the 2014 French Open.

 He had previously reached the quarterfinals of the 2008 French Open.

 Gulbis' career-high singles ranking is world No. 10, making him the only Latvian tennis player ever to be ranked inside the top 10 in ATP Singles Ranking.

 He achieved this in June 2014.
</item 1>

<item 2>
Vitas Gerulaitis
Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.

 In 1975, Gerulaitis won the men's doubles title at Wimbledon, partnering with Sandy Mayer.

 He won the men's singles title at one of the two Australian Open tournaments held in 1977 (Gerulaitis won the tournament that was held in December, while
Roscoe Tanner won the earlier January tournament).

 Gerulaitis also won two Italian Open titles, in 1977 and 1979, and the WCT Finals in Dallas, in 1978.
</item 2>

<item 3>
Heather Watson
Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.

 She is a former British No.1 and current British No. 2 behind Johanna Konta.
</item 3>

<item 4>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 4>

<item 5>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 5>

<item 6>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 6>

<item 7>
Sharon Walraven
Sharon Walraven (born 19 June 1970, Schaesberg) is a Dutch wheelchair tennis player.

 She became paraplegic at age 23 after complications following a fall while she was ice-skating.

 She has won seven Grand Slams doubles titles partnering compatriot Esther Vergeer.

 At the 2008 Paralympics in Beijing she won the Gold medal in the women's doubles competition.

 At the 2000 Paralympics in Sydney she won a Silver medal in the women's singles competition.

 Walraven has a highest ranking of No.2 in singles and No.1 in doubles.
</item 7>

<item 8>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 8>

<item 9>
1974 Stadthalle Open – Singles
Vitas Gerulaitis won in the final 6–4, 3–6, 6–3, 6–2 against Andrew Pattison.
</item 9>

<item 10>
1979 World Championship Tennis Finals – Singles
Vitas Gerulaitis was the defending champion but lost in the semifinals to Björn Borg.
</item 10>
=== END ITEM id=60959d9b3d718899 ===

=== ITEM id=efffac81b0d10af3 ===
Gold sentences and facts:
Sentence 0: The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.
  Fact 0.0: The presidency of Ronald Reagan began on January 20, 1981
  Fact 0.1: Ronald Reagan was inaugurated as President of the United States on January 20, 1981
  Fact 0.2: The presidency of Ronald Reagan ended on January 20, 1989
Sentence 1: George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice President of the United States from 1981 to 1989.
  Fact 1.0: George Herbert Walker Bush was born June 12, 1924
  Fact 1.1: George Herbert Walker Bush is an American politician
  Fact 1.2: George Herbert Walker Bush was the 41st President of the United States from 1989 to 1993
  Fact 1.3: George Herbert Walker Bush was the 43rd Vice President of the United States from 1981 to 1989

Retrieved items:
<item 1>
Presidency of Ronald Reagan
The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.

 Reagan, a Republican, took office as the 40th United States president following a landslide win over Democratic incumbent President Jimmy Carter in the 1980
presidential election.

 Reagan was succeeded by his vice president, George H. W. Bush, who won the 1988 presidential election with Reagan's support.
</item 1>

<item 2>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 2>

<item 3>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 3>

<item 4>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 4>

<item 5>
Reagan Democrat
A Reagan Democrat is a traditionally Democratic voter in the United States, referring especially to white working-class Rust Belt residents, who defected from
their party to support Republican President Ronald Reagan in either or both of the 1980 and 1984 elections as well as Republican Presidents George H. W. Bush in
the 1988 election and George W. Bush in either or both of the 2000 and 2004 elections.

 Part of this group also defected to Donald Trump, the Republican candidate in 2016, who won in Pennsylvania, Ohio, Michigan, Wisconsin, and Iowa, all states
that voted for Reagan but went Democratic in 2008 and 2012.
</item 5>

<item 6>
Ronald Reagan judicial appointment controversies
During President Ronald Reagan's presidency, he nominated at least twelve people for various federal appellate judgeship who were not confirmed.

 In some cases, the nominations were not processed by the Democratic-controlled Senate Judiciary Committee before Reagan's presidency ended, while in other
cases, nominees were rejected by the Senate Judiciary Committee or even blocked by unfriendly members of the Republican Party.

 Three of the nominees were renominated by Reagan's successor, President George H. W. Bush.

 Two of the nominees, Ferdinand Francis Fernandez and Guy G. Hurlbutt, were nominated after July 1, 1988, the traditional start date of the unofficial Thurmond
Rule during a presidential election year.

 Eight of the twelve seats eventually were filled by appointees of President George H. W. Bush.
</item 6>

<item 7>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 7>

<item 8>
First inauguration of Ronald Reagan
The first inauguration of Ronald Reagan as the 40th President of the United States was held on Tuesday, January 20, 1981, on the West Front of the United States
Capitol in Washington, D.C..

 This was the first inauguration to be held on the building's west side.

 The inauguration marked the commencement of the first four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.

 Chief Justice Warren E. Burger administered the presidential oath of office to Reagan, who placed his hand upon a family Bible given to him by his mother, open
to .

 Associate Justice Potter Stewart administered the vice presidential oath to Bush.
</item 8>

<item 9>
Second inauguration of Ronald Reagan
The second inauguration of Ronald Reagan as President of the United States was held privately on Sunday, January 20, 1985 in the Grand Foyer of the White House,
and publicly the following day, January 21, 1985 at the United States Capitol in Washington, D.C. The inauguration marked the commencement of the second
four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.
</item 9>

<item 10>
Elliott Abrams
Elliott Abrams (born January 24, 1948) is a former American diplomat, lawyer and political scientist who served in foreign policy positions for Presidents
Ronald Reagan and George W. Bush.

 Abrams was convicted of withholding information from Congress about the Iran–Contra affair while serving under Reagan, but was pardoned by President George H.
W. Bush.
</item 10>
=== END ITEM id=efffac81b0d10af3 ===
