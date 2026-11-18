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

This file has 13 items.

=== ITEM id=b105f5f2b6e41f94 ===
Gold sentences and facts:
Sentence 0: The Monastery of the Mother of God at the Spring (full name in Greek: , pr.
  Fact 0.0: The full Greek name of the Monastery of the Mother of God at the Spring is given, with the pronunciation abbreviation 'pr.'
Sentence 1:  "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an Eastern Orthodox sanctuary in Istanbul, Turkey.
  Fact 1.0: The Monastery of the Mother of God at the Spring has the name 'Moni tis Theotóku tis Pigis'.
  Fact 1.1: The Monastery of the Mother of God at the Spring has the Turkish name 'Balıklı Meryem Ana Rum Manastiri'.
  Fact 1.2: The Monastery of the Mother of God at the Spring is simply called Zoödochos Pege.
  Fact 1.3: Zoödochos Pege is Greek for 'Life-giving Spring' (Ζωοδόχος Πηγή).
  Fact 1.4: The Monastery of the Mother of God at the Spring is an Eastern Orthodox sanctuary.
  Fact 1.5: The Monastery of the Mother of God at the Spring is located in Istanbul, Turkey.
Sentence 2: Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.
  Fact 2.0: Hagia Irene is also called Hagia Eirene.
  Fact 2.1: Hagia Irene has the Ancient Greek name Ἁγία Εἰρήνη.
  Fact 2.2: A Byzantine form of the name is given for Hagia Irene, but the text of that form is cut off in the source.
  Fact 2.3: Hagia Irene's name means 'Holy Peace'.
  Fact 2.4: Hagia Irene has the Turkish name 'Aya İrini'.
  Fact 2.5: Hagia Irene is sometimes known also as Saint Irene.
  Fact 2.6: Hagia Irene is a Greek Eastern Orthodox church.
  Fact 2.7: Hagia Irene is located in the outer courtyard of Topkapı Palace.
  Fact 2.8: Topkapı Palace is in Istanbul, Turkey.

Retrieved items:
<item 1>
St. Mary's Church and Rectory (Iowa City, Iowa)
St. Mary's Catholic Church, also known as St. Mary of the Visitation Church, is a parish church of the Diocese of Davenport which is located at 228 E. Jefferson
St. in Iowa City, Iowa, United States.

 The church building and rectory are listed together on the National Register of Historic Places in 1980.

 They were both included as contributing properties in the Jefferson Street Historic District in 2004.

 The parish's first rectory, which is now a private home, is also listed on the National Register as St. Mary's Rectory.

 It is located a few blocks to the east of the present church location at 610 E. Jefferson St.
</item 1>

<item 2>
St. Mary's Church (Ballston Spa, New York)
St. Mary's Church is a Catholic parish located in Ballston Spa, New York.

 It is located within the Roman Catholic Diocese of Albany.

 Father Thomas J. Kelly is the current pastor.

 St. Mary's is the fourth oldest parish in the Diocese.

 St. Mary's of Ballston Spa is partnered with St. Mary's of Galway, after the pastor at St. Mary's of Galway died and no replacement was available.
</item 2>

<item 3>
St. Mary's Episcopal Church (Washington, D.C.)
St. Mary's Episcopal Church, also known as St. Mary's, Foggy Bottom or St. Mary's Chapel, is a historic Episcopal church located at 730 23rd Street, N.W. in the
Foggy Bottom neighborhood of Washington, D.C..

 On April 2, 1973, St. Mary's Episcopal Church was added to the National Register of Historic Places.
</item 3>

<item 4>
First Council of Constantinople
The First Council of Constantinople (Greek: Πρώτη σύνοδος της Κωνσταντινουπόλεως commonly known as Greek: Β΄ Οικουμενική , "Second Ecumenical"; Latin:
"Concilium Constantinopolitanum Primum" or Latin: "Concilium Constantinopolitanum A" ) was a council of Christian bishops convened in Constantinople in AD 381
by the Roman Emperor Theodosius I.

 This second ecumenical council, an effort to attain consensus in the church through an assembly representing all of Christendom, confirmed the Nicene Creed,
expanding the doctrine thereof to produce the Niceno–Constantinopolitan Creed, and dealt with sundry other matters.

 It met from May to July 381 in the Church of Hagia Irene and was affirmed as ecumenical in 451 at the Council of Chalcedon.
</item 4>

<item 5>
St Mary de Lode Church, Gloucester
St Mary de Lode Church, Archdeacon Street, Gloucester GL1 2QT, is a Church of England church immediately outside the grounds of Gloucester Cathedral.

 It is believed by some to be on the site of the first Christian church in Britain.

 The church is in the Diocese of Gloucester and Grade I listed by English Heritage.

 It has also been known as "St. Mary Before the Gate of St. Peter", "St. Mary Broad Gate" and "St. Mary De Port".
</item 5>

<item 6>
St. Mary's Church (Swormville, New York)
St. Mary's Catholic Church is a church located on the corner of Transit (New York State Route 78) and Stahley roads in Swormville, New York, in the United
States.

 Originally known as St. Mary’s Church of the Assumption at Transit, the building was constructed of about 260,000 bricks.

 John Nepomucene Neumann (later to be named St. John Neumann) started to visit Swormville, celebrating mass in local homes, barns and fields.

 In 1839 he instructed that a small house would be made to hold mass.

 Bishop John Timon established the first catholic church (St. Mary's) of the town of Clarence, New York.

 In 1861, Father Michael Schinabeak made plans for a church to be constructed.

 Construction began in 1862 and finished in fall of 1865.

 The official opening of the Church was in January 1866.

 In 2010 a new Church was constructed behind the old St. Mary's.
</item 6>

<item 7>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 7>

<item 8>
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
</item 8>

<item 9>
Church of St. Mary of the Spring (Istanbul)
The Monastery of the Mother of God at the Spring (full name in Greek: , pr.

 "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an
Eastern Orthodox sanctuary in Istanbul, Turkey.

 The present church, built in 1835, bears the same dedication as the shrine erected in this place between the end of the fifth and the beginning of the sixth
century.

 After several renovations, this building was destroyed in the first half of the fifteenth century by the Ottomans.

 The complex got its name from a nearby holy spring, reputed to have healing properties.

 For almost fifteen hundred years, this sanctuary has been one of the most important pilgrimage sites of Greek Orthodoxy.
</item 9>

<item 10>
Hagia Irene
Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek
Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.

 It is one of the few churches in Istanbul that has not been converted into a mosque.

 The Hagia Irene today operates as a museum and concert hall.
</item 10>
=== END ITEM id=b105f5f2b6e41f94 ===

=== ITEM id=a8c43cc44586af34 ===
Gold sentences and facts:
Sentence 0: The Basset Bleu de Gascogne (] ), also known as the Blue Gascony Basset, is a long-backed, short legged breed of dog of the hound type.
  Fact 0.0: The Basset Bleu de Gascogne is also known as the Blue Gascony Basset.
  Fact 0.1: The Basset Bleu de Gascogne is a long-backed, short-legged breed of dog.
  Fact 0.2: The Basset Bleu de Gascogne is a dog of the hound type.
Sentence 1:  A French native breed, it is rare outside of its homeland.
  Fact 1.0: The Basset Bleu de Gascogne is a French native breed.
  Fact 1.1: The Basset Bleu de Gascogne is rare outside of its homeland.
Sentence 2: The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.
  Fact 2.0: The Bulldog is a medium-sized breed of dog.
  Fact 2.1: The Bulldog is commonly referred to as the English Bulldog.
  Fact 2.2: The Bulldog is commonly referred to as the British Bulldog.

Retrieved items:
<item 1>
Ariegeois
The Ariegeois is a breed of dog from the "département" of Ariège in the Midi-Pyrenées region of southern France.

 It is a medium-sized pack-hunting scenthound deriving from crossing of Grand Bleu de Gascogne and Grand Gascon-Saintongeois hounds with local Briquet dogs.

 It is used both as a courser and for driving game to waiting guns.

 While most successful with hares, it is also used for hunting deer and boar.

 It is distinguished by its friendly nature with other hounds and affection for human companions.
</item 1>

<item 2>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 2>

<item 3>
Petit Bleu de Gascogne
The Petit Bleu de Gascogne (FCI No.31) Is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog.

 The Petit Bleu de Gascogne is not a small (petite) dog, the name comes from its use on small game.
</item 3>

<item 4>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 4>

<item 5>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 5>

<item 6>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 6>

<item 7>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 7>

<item 8>
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 8>

<item 9>
Basset Bleu de Gascogne
The Basset Bleu de Gascogne (] ), also known as the Blue Gascony Basset, is a long-backed, short legged breed of dog of the hound type.

 The breed originated in the Middle Ages, descended from the Grand Bleu de Gascogne.

 It nearly became extinct around the early 19th century; its salvation was attributed to one Alain Bourbon.

 A French native breed, it is rare outside of its homeland.

 It is recognized internationally by the Fédération Cynologique Internationale, in the UK by The Kennel Club, and by the United Kennel Club in the United
States.

 The "bleu" of its name is a reference to its coat which has a ticked appearance.
</item 9>

<item 10>
Bulldog
The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.

 Other scent-hound breeds include the Small Greek Domestic Dog, Irish Wolfhound, Bluetick Coonhound, Finnish Lapphund, and the Basset Hound.

 The Bulldog is a muscular, hefty dog with a wrinkled face and a distinctive pushed-in nose.

 The American Kennel Club (AKC), The Kennel Club (UK), and the United Kennel Club (UKC) oversee breeding records.

 Bulldogs were the fourth most popular purebreed in the US in 2007 according to the American Kennel Club.
</item 10>
=== END ITEM id=a8c43cc44586af34 ===

=== ITEM id=0bac5fe62558367c ===
Gold sentences and facts:
Sentence 0: Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.
  Fact 0.0: Rosario Castellanos Figueroa was born on 25 May 1925.
  Fact 0.1: Rosario Castellanos Figueroa died on 7 August 1974.
  Fact 0.2: Rosario Castellanos Figueroa was a Mexican poet.
  Fact 0.3: Rosario Castellanos Figueroa was a Mexican author.
Sentence 1: Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a series of novels beginning in 1908 with "Anne of Green Gables".
  Fact 1.0: Lucy Maud Montgomery was born on November 30, 1874.
  Fact 1.1: Lucy Maud Montgomery died on April 24, 1942.
  Fact 1.2: Lucy Maud Montgomery published as L.M. Montgomery.
  Fact 1.3: Lucy Maud Montgomery was a Canadian author.
  Fact 1.4: Lucy Maud Montgomery is best known for a series of novels.
  Fact 1.5: The series of novels began in 1908 with Anne of Green Gables.

Retrieved items:
<item 1>
Anne of Ingleside
Anne of Ingleside is a children's novel by Canadian author Lucy Maud Montgomery.

 It was first published in July 1939 by McClelland and Stewart (Toronto) and the Frederick A. Stokes Company (New York).

 It is the tenth of eleven books that feature the character of Anne Shirley, and Montgomery's final published novel.

 (Two novels that occur later in the "Anne" chronology were actually published years earlier.

 As well, the short story collection "The Blythes Are Quoted", written in 1941/42, but not published until 2009, concludes the Anne chronology.)
</item 1>

<item 2>
Anne of Green Gables
Anne of Green Gables is a 1908 novel by Canadian author Lucy Maud Montgomery (published as L. M. Montgomery).

 Written for all ages, it has been considered a children's novel since the mid-twentieth century.

 It recounts the adventures of Anne Shirley, an 11-year-old orphan girl who is mistakenly sent to Matthew and Marilla Cuthbert, a middle-aged brother and sister
who had intended to adopt a boy to help them on their farm in the fictional town of Avonlea on Prince Edward Island.

 The novel recounts how Anne makes her way with the Cuthberts, in school, and within the town.
</item 2>

<item 3>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 3>

<item 4>
L.M. Montgomery's Anne of Green Gables
L.M. Montgomery's Anne of Green Gables is a Canadian television film based on Lucy Maud Montgomery's novel of the same name.

 It first aired on YTV on February 15, 2016 and starred Ella Ballentine, Martin Sheen and Sara Botsford.

 Montgomery's granddaughter, Kate Macdonald Butler, was one of the film's executive producers.

 The film's world premiere was held February 2, 2016 at the Canadian Museum of History.
</item 4>

<item 5>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 5>

<item 6>
Anne Shirley
Anne Shirley is a fictional character introduced in the 1908 novel "Anne of Green Gables" by Lucy Maud Montgomery.

 Montgomery wrote in her journal that the idea for Anne's story came from relatives who, planning to adopt an orphaned boy, received a girl instead.

 Anne Shirley's appearance was inspired by a photograph which Montgomery clipped from the Metropolitan Magazine and kept, unaware of the model's identity as the
1900s Gibson Girl Evelyn Nesbit.
</item 6>

<item 7>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 7>

<item 8>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 8>

<item 9>
Emily's Quest
Emily's Quest is a novel and the last of the "Emily" trilogy by Lucy Maud Montgomery.

 After finishing "Emily Climbs", Montgomery suspended writing "Emily's Quest" and published "The Blue Castle"; she resumed writing and published in 1927.
</item 9>

<item 10>
Rosario Castellanos
Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.

 Along with the other members of the Generation of 1950 (the poets who wrote following the Second World War, influenced by César Vallejo and others), she was
one of Mexico's most important literary voices in the last century.

 Throughout her life, she wrote eloquently about issues of cultural and gender oppression, and her work has influenced feminist theory and cultural studies.

 Though she died young, she opened the door of Mexican literature to women, and left a legacy that still resonates today.
</item 10>
=== END ITEM id=0bac5fe62558367c ===

=== ITEM id=6f7eaf4573e25e12 ===
Gold sentences and facts:
Sentence 0: Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the United States.
  Fact 0.0: Pago Pago International Airport has IATA code PPG.
  Fact 0.1: Pago Pago International Airport has ICAO code NSTU.
  Fact 0.2: Pago Pago International Airport has FAA LID PPG.
  Fact 0.3: Pago Pago International Airport is also known as Tafuna Airport.
  Fact 0.4: Pago Pago International Airport is a public airport.
  Fact 0.5: Pago Pago International Airport is located 7 miles (11.3 km) southwest of the central business district of Pago Pago.
  Fact 0.6: Pago Pago International Airport is located in the village and plains of Tafuna.
  Fact 0.7: Tafuna is on the island of Tutuila.
  Fact 0.8: Tutuila is in American Samoa.
  Fact 0.9: American Samoa is an unincorporated territory of the United States.
Sentence 1: Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business district of Hoonah, Alaska.
  Fact 1.0: Hoonah Airport has IATA code HNH.
  Fact 1.1: Hoonah Airport has ICAO code PAOH.
  Fact 1.2: Hoonah Airport has FAA LID HNH.
  Fact 1.3: Hoonah Airport is a state-owned public-use airport.
  Fact 1.4: Hoonah Airport is located one nautical mile (2 km) southeast of the central business district of Hoonah.
  Fact 1.5: Hoonah is in Alaska.

Retrieved items:
<item 1>
Pago Pago International Airport
Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of
the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the
United States.
</item 1>

<item 2>
Pan Am Flight 806
Pan Am Flight 806 was an international scheduled flight from Auckland, New Zealand, to Los Angeles, California, with intermediate stops at Pago Pago, American
Samoa and Honolulu, Hawaii.

 On January 30, 1974, the Boeing 707 "Clipper Radiant" crashed on approach to Pago Pago International Airport, killing 87 passengers and ten crew members.
</item 2>

<item 3>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 3>

<item 4>
Tutuila
Tutuila is the largest and the main island of American Samoa in the archipelago of Samoan Islands.

 It is the third largest island in the Samoan Islands chain of the Central Pacific located roughly 4000 km northeast of Brisbane, Australia and over 1200 km
northeast of Fiji.

 It contains a large, natural harbor, Pago Pago Harbor, where Pago Pago, the capital of American Samoa is situated.

 Pago Pago International Airport is also located on Tutuila island.

 Its land expanse is about 68% of the total land area of American Samoa and with 56,000 people accounts for 95% of its population.

 The island has six terrestrial and three marine ecosystems.
</item 4>

<item 5>
Veterans Memorial Stadium (Pago Pago)
Veterans Memorial Stadium is a football stadium located in Pago Pago Park, in Pago Pago, American Samoa.

 The 10,000 capacity venue is one of the smallest stadiums in Oceania, and serves as American Samoa's national stadium.

 It is the home venue of the American Samoa national football team, hosting all of their home games.

 It is currently used mostly for matches in various football codes, such as soccer, rugby league, and the territory's most popular code, American football.
</item 5>

<item 6>
Inter Island Airways
Inter Island Airways (also known as "Inter Island Air") is a South Pacific regional airline based in Pago Pago, American Samoa.

 Inter Island Airways operates passenger and cargo flights in and between American Samoa, Independent Samoa and to neighboring Pacific island countries.

 Its main base of operations is at Pago Pago International Airport.
</item 6>

<item 7>
Tutuila, American Samoa’s largest island and home to 95% of its population, contains Pago Pago, the territorial capital and harbor, and Pago Pago International
Airport at Tafuna. Nearby villages include Nuʻuuli and Tafuna. The 10,000-seat Veterans Memorial Stadium is the national stadium and hosts several football
codes. Inter Island Airways serves Samoa and neighboring Pacific islands from Pago Pago. NOAA station WNG710 broadcasts weather and hazard
</item 7>

<item 8>
Pago Pago
Pago Pago ( ; Samoan: ] ) is the territorial capital of American Samoa.

 It is on the main island of American Samoa, Tutuila.

 The territory is served by Pago Pago International Airport at Tafuna, some 8 miles south west of Pago Pago.

 Tourism, entertainment, food, and tuna canning are its main industries.
</item 8>

<item 9>
WNG710
WNG710 (sometimes referred to as Pago Pago All Hazards) is a NOAA Weather Radio station that serves the entire United States territory island of American Samoa.

 It is programmed from the National Weather Service Forecast Office in Pago Pago.

 The station broadcasts weather and hazard information for the Eastern, Manu'a, and Western Districts, which are the three political divisions of American
Samoa.

 WNG710 has a sister station, WZ2529 at Mt. Olotele.

 Its transmitter is located at the opposite side of the island.
</item 9>

<item 10>
Tafuna, American Samoa
Tafuna' is a village on the east coast of Tutuila Island, American Samoa.

 It is located on a peninsula a mile north of Pago Pago International Airport and one mile south of Nu'uuli, American Samoa.
</item 10>
=== END ITEM id=6f7eaf4573e25e12 ===

=== ITEM id=ce6a6db224b45999 ===
Gold sentences and facts:
Sentence 0: Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.
  Fact 0.0: Bicycling is a cycling brand.
  Fact 0.1: Bicycling is published by Rodale, Inc. in Emmaus, Pennsylvania.
Sentence 1: National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural affairs.
  Fact 1.0: National Review (NR) is an American magazine.
  Fact 1.1: National Review is semi-monthly.
  Fact 1.2: National Review is conservative.
  Fact 1.3: National Review is an editorial magazine.
  Fact 1.4: National Review focuses on news and commentary pieces on political affairs.
  Fact 1.5: National Review focuses on news and commentary pieces on social affairs.
  Fact 1.6: National Review focuses on news and commentary pieces on cultural affairs.

Retrieved items:
<item 1>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 1>

<item 2>
Charles C. W. Cooke
Charles C. W. Cooke (born November 4, 1984) is the editor of "National Review Online".

 He took the role over after Rich Lowry stepped down in June, 2016 (Lowry remains the editor-in-chief of "National Review").

 Cooke is the author of "The Conservatarian Manifesto" and a frequent guest on HBO's "Real Time with Bill Maher".

 In addition to "National Review", he has written for the "New York Times", the "Washington Post", and the "Los Angeles Times".

 Along with Kevin D. Williamson, he hosts the popular "Mad Dogs and Englishmen" podcast.
</item 2>

<item 3>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 3>

<item 4>
Review
A review is an evaluation of a publication, service, or company such as a movie (a movie review), video game (video game review), musical composition (music
review of a composition or recording), book (book review); a piece of hardware like a car, home appliance, or computer; or an event or performance, such as a
live music concert, play, musical theater show, dance show, or art exhibition.

 In addition to a critical evaluation, the review's author may assign the work a rating to indicate its relative merit.

 More loosely, an author may review current events, trends, or items in the news.

 A compilation of reviews may itself be called a review. "

The New York Review of Books", for instance, is a collection of essays on literature, culture, and current affairs. "

National Review", founded by William F. Buckley, Jr., is an influential conservative magazine, and "Monthly Review" is a long-running socialist periodical.
</item 4>

<item 5>
National Review
National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural
affairs.

 The magazine was founded by the author William F. Buckley Jr. in 1955.

 It is currently edited by Rich Lowry.
</item 5>

<item 6>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 6>

<item 7>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 7>

<item 8>
National Review Board
The National Review Board (full name: National Review Board for the Protection of Children and Young People) is a committee created in 2002 by the United States
Conference of Catholic Bishops in order to monitor the implementation of the Charter for the Protection of Children and Young People in the wake of the clerical
abuse scandal in the United States.

 The Board was also charged with investigating the scandal, which it did in part by commissioning the John Jay College to conduct a survey of Church records in
order to define the nature and scope of the abuse committed by priests over the period between 1950 and 2002.

 The results of that survey were released in 2004 in what has come to be known as the "John Jay Report" or "Nature and Scope report."

 In parallel with the John Jay College survey the Board conducted interviews with a variety of people, both inside and outside the Church, who were well placed
to comment on the scandal, and on the basis of these interviews prepared a more broad-ranging report of its own.
</item 8>

<item 9>
Kevin D. Williamson
Kevin Daniel Williamson (born September 18, 1972) is the roving correspondent for "National Review."

 He is also the theater critic for "The New Criterion".

 He was previously deputy managing editor at "National Review".
</item 9>

<item 10>
National Review Board
Whereas the John Jay College report was (as intended) primarily a factual summary of the data collected in the College’s survey of Church records, the Board’s
own report sought to interpret these data and its other findings in order to explain why the “epidemic” of clerical abuse had occurred and to identify the
appropriate steps to avoid any repetition.
</item 10>
=== END ITEM id=ce6a6db224b45999 ===

=== ITEM id=a62b4258c1a67890 ===
Gold sentences and facts:
Sentence 0: Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.
  Fact 0.0: Wenling is a coastal county-level city.
  Fact 0.1: Wenling is in the municipal region of Taizhou.
  Fact 0.2: Taizhou is in southeastern Zhejiang province.
  Fact 0.3: Wenling is in China.
  Fact 0.4: In the Wenling dialect, Wenling is called Uen-lin Zy.
Sentence 1: Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.
  Fact 1.0: Ruili is a county-level city of Dehong Prefecture.
  Fact 1.1: Ruili is in the west of Yunnan province.
  Fact 1.2: Yunnan province is in the People's Republic of China.
  Fact 1.3: Ruili's Burmese name is ရွှေလီ.
  Fact 1.4: Ruili's Thai name is เมืองมาว.

Retrieved items:
<item 1>
Ruili Border Economic Cooperation Zone
Ruili Border Economic Cooperation Zone (RLBECZ) is a Chinese State Council-approved Industrial Park based in Ruili City, Dehong Prefecture, Yunnan, China,
founded in 1992 and was established to promote trade between China and Myanmar.

 The zone connects Ruili with Man-wing City.
</item 1>

<item 2>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 2>

<item 3>
Wanding Border Economic Cooperation Zone
Wanding Border Economic Cooperation Zone (WTBECZ) is a Chinese State Council-approved Industrial Park based in Wanding Town, Ruili City, Dehong Prefecture,
Yunnan, China, founded in 1992 and was established to promote trade between China and Myanmar.

 The zone spans 6 km sq. and is focuses on developing trading, processing, agriculture resources and tourism.
</item 3>

<item 4>
Dali–Ruili Railway
The Dali–Ruili Railway or Darui Railway (), is a single-track electrified railroad under construction in Yunnan Province of Southwest China.

 The line is slated to run 336.39 km from Dali in central Yunnan to Ruili in southwestern Yunnan on the border with Myanmar.

 The line traverses rugged terrain, and bridges and tunnels will account for 75% of the total track length, including a 36-km tunnel through the Gaoligong
Mountains.

 Construction began in May 2011 and is scheduled to take six years.

 Cities and towns along route include Dali, Yangbi Yi Autonomous County, Yongping County, Baoshan, Mang City (also known as Mangshi or Dehong) and Ruili.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 6>

<item 7>
G56 Hangzhou–Ruili Expressway
The Hangzhou–Ruili Expressway (), commonly referred to as the "Hangrui Expressway" () is an expressway in China that connects the cities of Hangzhou, Zhejiang,
and Ruili, Yunnan, a city on the border with Burma.

 When complete, it will be 2935 km in length.
</item 7>

<item 8>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 8>

<item 9>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 9>

<item 10>
Ruili
Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.

 It is a major border crossing between China and Myanmar, with the town of Muse located across the border.
</item 10>
=== END ITEM id=a62b4258c1a67890 ===

=== ITEM id=90c8aad53dffaedf ===
Gold sentences and facts:
Sentence 0: Park Sung-woong (born January 9, 1973) is a South Korean actor.
  Fact 0.0: Park Sung-woong was born on January 9, 1973.
  Fact 0.1: Park Sung-woong is a South Korean actor.
Sentence 1:  Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a serial killer in "The Deal" (2015).
  Fact 1.0: Park Sung-woong made his acting debut in 'No. 3' in 1997.
  Fact 1.1: Park Sung-woong has starred in several movies and television series.
  Fact 1.2: Park Sung-woong notably starred as a gangster in 'New World' (2013).
  Fact 1.3: Park Sung-woong notably starred as a serial killer in 'The Deal' (2015).
Sentence 2: The Deal (; lit.
  Fact 2.0: The Deal's literal translation is given in the next sentence, after the 'lit.' marker.
Sentence 3:  "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
  Fact 3.0: The literal translation of The Deal is 'Murder Request'.
  Fact 3.1: The Deal is a 2015 South Korean crime thriller film.
  Fact 3.2: The Deal was directed by Son Yong-ho.

Retrieved items:
<item 1>
Deep Trap
Deep Trap () (previously known as Exchange) is a 2015 South Korean crime thriller film starring Ma Dong-seok, Jo Han-sun and Kim Min-kyung, and directed by Kwon
Hyung-jin.

 Based on an actual SNS crime, it depicts the terror experienced by a married couple while on a trip to an isolated island.

 It won Best Film Award in the Orient Express section at the Fantasporto in 2016.
</item 1>

<item 2>
Office (2015 South Korean film)
Office (오피스 ) is 2015 South Korean slasher-thriller film directed by Hong Won-chan and starring Go Ah-sung and Park Sung-woong.

 It is about a detective trying to figure out why a mild-mannered man has killed his family and is targeting his co-workers.

 The film premiered at the Midnight Screenings section of the 2015 Cannes Film Festival.
</item 2>

<item 3>
The Boys Who Cried Wolf
The Boys Who Cried Wolf (), also known as The Shepherd, is a 2015 South Korean crime thriller drama film.

 Written and directed by Kim Jin-hwang in his first feature-length for his Korean Academy of Film Arts (KAFA)'s final year undergraduate film project, it
depicts how a former stage actor becomes involved in a murder case.
</item 3>

<item 4>
The Swindlers (2017 film)
The Swindlers is an upcoming South Korean crime drama film directed by Jang Chang-won.

 The film stars Hyun Bin, Yoo Ji-tae, Bae Seong-woo, Park Sung-woong, Nana and Ahn Se-ha.
</item 4>

<item 5>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 5>

<item 6>
A Company Man
A Company Man () is a 2012 South Korean action thriller film starring So Ji-sub, Lee Mi-yeon, Kwak Do-won and Kim Dong-jun.

 It is about a hitman who find himself targeted by his ex-employers after he falls in love with a single mother and quits his job.
</item 6>

<item 7>
Fatal Intuition
Fatal Intuition (; lit.

 "It's Him") is a 2015 South Korean crime thriller film written and directed by Yun Jun-hyeong.
</item 7>

<item 8>
Park Sung-woong
Park Sung-woong (born January 9, 1973) is a South Korean actor.

 Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a
serial killer in "The Deal" (2015).
</item 8>

<item 9>
A Violent Prosecutor
A Violent Prosecutor is a 2016 South Korean crime film directed by Lee Il-hyung, produced by Guk Su Ran and starring Hwang Jung-min, Kang Dong-won, Lee Sung-min
and Park Sung-woong.

 It was released in South Korea on February 3, 2016 by Showbox.
</item 9>

<item 10>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 10>
=== END ITEM id=90c8aad53dffaedf ===

=== ITEM id=a4f0f8380ded137e ===
Gold sentences and facts:
Sentence 0: Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
  Fact 0.0: Grantley Jonathan "Grant" Nicholas was born on 12 November 1967.
  Fact 0.1: Grant Nicholas is a Welsh musician.
  Fact 0.2: Grant Nicholas is best known as the lead singer of the rock band Feeder.
  Fact 0.3: Grant Nicholas is best known as the lead guitarist of the rock band Feeder.
Sentence 1: Danny Shirley (born August 12, 1956) is an American country music singer.
  Fact 1.0: Danny Shirley was born on August 12, 1956.
  Fact 1.1: Danny Shirley is an American country music singer.

Retrieved items:
<item 1>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 1>

<item 2>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 2>

<item 3>
Nicholas Mele
Nicholas Mele is an actor who has starred in many movies and on television.

 His first movie role was in the 1976 movie "The Ritz".

 Other movie roles include "Some Kind of Hero" (1982) and "Young Doctors in Love" (1982).

 His most well-known film role was in the 1988 hit horror movie "" as Dennis Johnson.

 A year later, he reprised his role in the hit sequel "".

 His most recent movie is the 2003 film "The Great Gabble".
</item 3>

<item 4>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 4>

<item 5>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 5>

<item 6>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 6>

<item 7>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 7>

<item 8>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 8>

<item 9>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 9>

<item 10>
Yorktown Heights (album)
Yorktown Heights was the only solo album by Feeder frontman Grant Nicholas.

 The album was released on 11 August 2014 in the United Kingdom and recorded during 2013, being a year after Nicholas announced at Feeder's final show of 2012
at the Brixton Academy that the band would be taking a break.

 It was with the first airplay of ""Soul Mates"" on the BBC Radio 2 show "The Dermot O'Leary Show" on 31 May 2014, that Grant's solo career made its radio
debut.
</item 10>
=== END ITEM id=a4f0f8380ded137e ===

=== ITEM id=fdaeff2237b6a965 ===
Gold sentences and facts:
Sentence 0: Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist Tayeb Salih.
  Fact 0.0: Season of Migration to the North is a classic post-colonial Sudanese novel.
  Fact 0.1: Season of Migration to the North is written by the novelist Tayeb Salih.
  Fact 0.2: The Arabic title of Season of Migration to the North is موسم الهجرة إلى الشمال, transliterated Mawsim al-Hiǧra ilā ash-Shamāl.
Sentence 1:  The novel is a counter narrative to "Heart of Darkness".
  Fact 1.0: Season of Migration to the North is a counter narrative to Heart of Darkness.
Sentence 2: Heart of Darkness (1899) is a novella by Polish-British novelist Joseph Conrad, about a voyage up the Congo River into the Congo Free State, in the heart of Africa, by the story's narrator Charles Marlow.
  Fact 2.0: Heart of Darkness is dated 1899.
  Fact 2.1: Heart of Darkness is a novella.
  Fact 2.2: Heart of Darkness is by Polish-British novelist Joseph Conrad.
  Fact 2.3: Heart of Darkness is about a voyage up the Congo River into the Congo Free State.
  Fact 2.4: The voyage in Heart of Darkness goes into the heart of Africa.
  Fact 2.5: The narrator of Heart of Darkness is Charles Marlow.

Retrieved items:
<item 1>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 1>

<item 2>
Season of Migration to the North
Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist
Tayeb Salih.

 Originally published in Arabic in 1966, it has since been translated into more than twenty languages.

 Salih was fluent in both English and Arabic, but chose to pen this novel in Arabic.

 The English translation was published in 1969 as part of the influential Heinemann African Writers Series.

 The novel is a counter narrative to "Heart of Darkness".

 It was described by Edward Said as one of the six great novels in Arabic literature.

 In 2001 it was selected by a panel of Arab writers and critics as the most important Arab novel of the twentieth century.
</item 2>

<item 3>
First North Americans
First North Americans are a series of historical fiction novels published by Tor and written by husband and wife co-authors W. Michael Gear & Kathleen O'Neal
Gear.

 The series, which began with 1990's "People of the Wolf", explores various civilizations and cultures in prehistoric North America.

 It is comparable to Jean M. Auel's "Earth's Children" series, set in prehistoric Europe, but each of its books focuses on a different time period, location,
and set of characters.

 The first four novels form a coherent, more or less linear narrative, from the initial migration of Siberian peoples into what is now Canada and Alaska (People
of the Wolf) through the florescence of the Mississippian semi-urban mound-building culture, considered the "high-water mark" of North American pre-Columbian
civilization, around 1000 AD.

 The remaining novels cover a wide variety of times and settings, in no particular order, ranging from tropical Florida in the 6th millennium BC to the Chaco
Empire of the Southwest in the 13th century AD.
</item 3>

<item 4>
Graph query row:
work: Heart of Darkness
relationship: CREATED_BY
connected_entity: Joseph Conrad
</item 4>

<item 5>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 5>

<item 6>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 6>

<item 7>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 7>

<item 8>
First North Americans
The novels take into account new developments in North American archaeology such as the discovery of Kennewick Man and the development of the coastal-route
model as a possible alternative or supplement to overland migration across Beringia.
</item 8>

<item 9>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 9>

<item 10>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 10>
=== END ITEM id=fdaeff2237b6a965 ===

=== ITEM id=3c496acb78b136d8 ===
Gold sentences and facts:
Sentence 0:  They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.
  Fact 0.0: This team is most well known for including Sugar Rashad Evans.
  Fact 0.1: Sugar Rashad Evans is a former UFC light heavyweight champion.
  Fact 0.2: This team is most well known for including Vitor Belfort.
  Fact 0.3: Vitor Belfort is a former UFC light heavyweight champion.
  Fact 0.4: This team is most well known for including Eddie Alvarez.
  Fact 0.5: Eddie Alvarez is a former UFC lightweight champion.
Sentence 1: Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
  Fact 1.0: Vitor Vieira Belfort was born on April 1, 1977.
  Fact 1.1: Vitor Vieira Belfort is a Brazilian mixed martial artist.
  Fact 1.2: Vitor Vieira Belfort is currently competing in the Middleweight division of the Ultimate Fighting Championship.

Retrieved items:
<item 1>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 1>

<item 2>
David Branch (fighter)
David Branch (born September 26, 1981) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2007, he has also formerly competed for the World Series of Fighting, Bellator, Shark Fights, and Titan FC.

 Branch was the inaugural World Series of Fighting Light Heavyweight and Middleweight Champion.

 He is currently ranked as the #9 Middleweight in the world according to Sherdog.
</item 2>

<item 3>
Luke Barnatt
Luke Barnatt (born 13 April 1988) is an English mixed martial artist who fought as a Middleweight for the Ultimate Fighting Championship.

 He was a member of FX's , and has also competed for BAMMA and UCMMA in England.

 At 6' 6" Barnatt is one of the tallest fighters currently competing in his weight class.

 Barnatt received his nickname "Bigslow" from brothers John Maguire and Tommy Maguire, for his size and being slow in response to their jokes and insults.

 Barnatt competes in the middleweight division of the Absolute Championship Berkut.
</item 3>

<item 4>
Luke Zachrich
Luke Zachrich (born October 1, 1981) is an American mixed martial artist and former professional boxer who formerly competed in the Middleweight division of the
Ultimate Fighting Championship A professional mixed martial arts competitor since 2006, Zachrich was a member of Team Forrest on the seventh season of The
Ultimate Fighter, and has also fought for King of the Cage, Xtreme Fighting Organization, Bellator, and Ultimate Victory Challenge, the latter of which he was
the inaugural Middleweight Champion.
</item 4>

<item 5>
Sam Alvey
Samuel Alvey (born May 6, 1986) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.

 A professional MMA competitor since 2008, Alvey has also formerly competed for Bellator, King of the Cage, the MFC, and was a contestant on .

 As of March 2017, Alvey is #14 in the official UFC Middleweight Rankings.
</item 5>

<item 6>
Blackzilians
The Blackzilians is a defunct professional team of fighters in mixed martial arts, boxing, kickboxing, amateur wrestling and catch wrestling located in South
Florida.

 They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC
lightweight champion Eddie Alvarez.
</item 6>

<item 7>
Rashad Evans
Rashad Anton Evans (born September 25, 1979) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2003, Evans was the Heavyweight winner of The Ultimate Fighter 2 and is also a former UFC Light Heavyweight Champion.
</item 7>

<item 8>
Kelvin Gastelum
Kelvin Gastelum (born October 24, 1991) is an American professional mixed martial artist who is currently signed to the Ultimate Fighting Championship (UFC).

 He was the winner of "The Ultimate Fighter 17".

 Gastelum is currently competing as a middleweight and is ranked #8 in the UFC official middleweight rankings.
</item 8>

<item 9>
Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 He is the former UFC Light Heavyweight Champion, UFC 12 Heavyweight Tournament Champion, and the Cage Rage World Light Heavyweight Champion.

 As of June 16, 2017, he is the #11 contender in official UFC Middleweight rankings.
</item 9>

<item 10>
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 10>
=== END ITEM id=3c496acb78b136d8 ===

=== ITEM id=5e7473f4d2bc490c ===
Gold sentences and facts:
Sentence 0: The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil War.
  Fact 0.0: The Seven Days Battles were a series of six major battles.
  Fact 0.1: The Seven Days Battles were fought over the seven days from June 25 to July 1, 1862.
  Fact 0.2: The Seven Days Battles were fought near Richmond, Virginia.
  Fact 0.3: The Seven Days Battles took place during the American Civil War.
Sentence 1: The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.
  Fact 1.0: The Battle of Manila took place from February 3, 1945 to March 3, 1945.
  Fact 1.1: The Battle of Manila was a major battle of the Philippine campaign of 1944-45.
  Fact 1.2: The Battle of Manila took place during the Second World War.

Retrieved items:
<item 1>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 1>

<item 2>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
</item 2>

<item 3>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 3>

<item 4>
Diwa de Leon
Diwa de Leon is a composer, arranger and musician based in Manila, Philippines.

 He makes music mainly for television shows, films, and stage productions.

 His most notable work is his arrangement of GMA Network's "Survivor Philippines" theme music.

 His songs included in "Emir".

 He won the Cinema One Originals Best Musical Score award for his work on the film "Kolorete" and Best Music Score for "Kamera Obskura" at the 8th Cinemalaya
Independent Film Festival.

 He is also a founding member of the world music group Makiling.

 His first mainstream film project is Mamarazzi produced by Regal Films.
</item 4>

<item 5>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 5>

<item 6>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 6>

<item 7>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 7>

<item 8>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 8>

<item 9>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 9>

<item 10>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 10>
=== END ITEM id=5e7473f4d2bc490c ===

=== ITEM id=a901eea5afedec13 ===
Gold sentences and facts:
Sentence 0:  The S70 was essentially a facelifted 850 saloon.
  Fact 0.0: The S70 was essentially a facelifted 850 saloon.
Sentence 1: The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.
  Fact 1.0: The Volvo 850 is a compact executive car.
  Fact 1.1: The Volvo 850 was produced by the Swedish manufacturer Volvo Cars.
  Fact 1.2: The Volvo 850 was produced from 1991 to 1996.
Sentence 2:  The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.
  Fact 2.0: The Volvo 850 was the first front-wheel drive vehicle from Volvo to be sold in North America.
  Fact 2.1: The Volvo 850 was the first all-wheel drive Volvo.

Retrieved items:
<item 1>
Volvo P2 platform
The Volvo P2 platform is a global full-size unibody automobile platform developed and produced by Volvo.

 It is designed for single wheelbases and is adaptable to front- or all wheel drive configurations.

 It was developed by the automaker before its 1999 acquisition by Ford Motor Company, and debuted with the 1998 Volvo S80.
</item 1>

<item 2>
Volvo 164
The Volvo 164 is a 4-door, 6-cylinder luxury sedan unveiled by Volvo at the Paris Motor Show early in October 1968 and first sold as a 1969 model.

 46,008 164s were built before the car was superseded by the 264 in 1975.

 The 164 was Volvo's first venture into the luxury segment since the end of PV 60 production in 1950, and was the first six-cylinder Volvo since the PV800 last
produced in 1958.
</item 2>

<item 3>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 3>

<item 4>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 4>

<item 5>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 5>

<item 6>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 6>

<item 7>
Volvo 480
The Volvo 480 is a compact car that was produced in Born, Netherlands, by Volvo from 1986 to 1995.

 It was the first front-wheel drive car made by the automaker.

 The 480 was available in only one body style on a automobile platform related to the Volvo 440/460 five door hatchback and four door sedan models.
</item 7>

<item 8>
Volvo 440/460
The Volvo 440 and 460 are versions of a small family car produced by the Swedish manufacturer Volvo between 1987 and 1996.

 The 440 was introduced in 1987, whilst the 460 followed in 1989.

 They were built at the NedCar factory in Born, The Netherlands.

 The 440 was a five door front wheel drive hatchback and the 460 a saloon.

 They shared many components with the already successful Volvo 480, including floorpan, front and rear suspension, engines, transmission and braking systems.
</item 8>

<item 9>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 9>

<item 10>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 10>
=== END ITEM id=a901eea5afedec13 ===

=== ITEM id=48b21e1bb8ef16e4 ===
Gold sentences and facts:
Sentence 0:  The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
  Fact 0.0: The women's events include the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park.
  Fact 0.1: The women's events include the Roxy Pro at Sunset Beach.
  Fact 0.2: The women's events include the Billabong Pro Maui at Honolua Bay, Maui.
Sentence 1: Haleʻ iwa is a North Shore community and census-designated place (CDP) in the Waialua District of the island of Oʻ ahu, City and County of Honolulu.
  Fact 1.0: Haleʻiwa is a North Shore community.
  Fact 1.1: Haleʻiwa is a census-designated place (CDP).
  Fact 1.2: Haleʻiwa is in the Waialua District of the island of Oʻahu.
  Fact 1.3: Haleʻiwa is in the City and County of Honolulu.

Retrieved items:
<item 1>
Andy Irons
Philip Andrew Irons (July 24, 1978November 2, 2010) was an American professional surfer.

 Irons learned to surf on the dangerous and shallow reefs of the North Shore in Kauai, Hawaii.

 Over the course of his professional career, he won three world titles (2002, 2003, 2004), three Quiksilver Pro France titles (2003, 2004, 2005), two Rip Curl
Pro Search titles (2006 and 2007) and 20 elite tour victories including the Vans Triple Crown of Surfing four times from 2002-2006.

 On September 3, 2010 he won the Billabong Pro Teahupoo in Tahiti.

 He and his family hosted the Annual Irons Brothers Pinetrees Classic, a contest for youngsters.

 The Governor of Hawaii declared February 13 forever "Andy Irons Day".

 He is the only surfer to have won a title at every venue on the ASP calendar.
</item 1>

<item 2>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 2>

<item 3>
World Match Racing Tour
The World Match Racing Tour is an annual match racing tour and World Championship in the match racing event of sailing.

 The World Match Racing Tour has "special event"-status from the International Sailing Federation.

 The Tour awards over US$1.5 million in prize money, with points awarded at each event culminating in the crowning of the ISAF Match Racing World Champion.

 A US$500,000 overall prize pool for the Championship is now on offer.
</item 3>

<item 4>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 4>

<item 5>
Ocean City Open
The Ocean City Open is a defunct tennis tournament that was played on the Grand Prix tennis circuit in 1977.

 The event was held in Ocean City, Maryland.

 Vitas Gerulaitis won the singles title while Alex Metreveli and Bill Scanlon partnered to win the doubles title.
</item 5>

<item 6>
Reef Hawaiian Pro
The Reef Hawaiian Pro is a surfing event held annually on the North Shore, Oahu, Hawaii.

 It is currently the first leg of the Triple Crown of Surfing.

 Surfers compete for a share of $135,000 (men) or $35,000 (women) prize money.

 Since it carries a 6-star rating, the event is critical for those who want to qualify for the ASP World Tour.
</item 6>

<item 7>
Triple Crown of Surfing
The Vans Triple Crown of Surfing is a Hawaiian specialty series of professional surfing events, offering three events to men and three events to women.

 For the men, those events are the Reef Hawaiian Pro at Haleiwa Ali'i Beach Park; the O'Neill World Cup of Surfing at Sunset Beach; and the Billabong Pipeline
Masters at the Banzai Pipeline.

 The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
</item 7>

<item 8>
2012 Monterrey Open
The 2012 Monterrey Open was a women's tennis tournament played on outdoor hard courts.

 It was the 4th edition of the Monterrey Open and was an International tournament on the 2012 WTA Tour.

 It took place at the Sierra Madre Tennis Club in Monterrey, Mexico, from 20 to 26 February.

 Tímea Babos won the singles title.
</item 8>

<item 9>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 9>

<item 10>
Graph query row:
series: Vans Triple Crown of Surfing
event: Vans Hawaiian Pro
held_on: None
location: Haleiwa Ali'i Beach Park
event_location: None
</item 10>
=== END ITEM id=48b21e1bb8ef16e4 ===
