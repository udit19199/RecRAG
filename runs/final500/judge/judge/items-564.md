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

=== ITEM id=49956d68f0140e64 ===
Gold sentences and facts:
Sentence 0: The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.
  Fact 0.0: The Dundee Canal was an industrial canal.
  Fact 0.1: The Dundee Canal was in Clifton and Passaic.
  Fact 0.2: Clifton and Passaic are in Passaic County, New Jersey.
Sentence 1:  It was built between 1858 and 1861 and ran parallel to the Passaic River.
  Fact 1.0: The Dundee Canal was built between 1858 and 1861.
  Fact 1.1: The Dundee Canal ran parallel to the Passaic River.
Sentence 2:  It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.
  Fact 2.0: It runs parallel to the coast.
  Fact 2.1: It is separated from the Atlantic Ocean by barrier beaches.
  Fact 2.2: The barrier beaches include Palm Beach Island.
Sentence 3:  The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
  Fact 3.0: The lagoon is connected to the Atlantic Ocean.
  Fact 3.1: The lagoon is connected to the Atlantic Ocean by two inlets.
  Fact 3.2: The two inlets are permanent.
  Fact 3.3: The two inlets are man-made.

Retrieved items:
<item 1>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 1>

<item 2>
Lake Worth Inlet
The Palm Beach Inlet, also known as the Lake Worth Inlet is an artificial cut through a barrier island connecting the northern part of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 It is bordered by the town of Palm Beach on the south, and by the town of Palm Beach Shores to the north.

 The inlet is also the entrance channel for the Port of Palm Beach.

 Its coordinates are .
</item 2>

<item 3>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 3>

<item 4>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 4>

<item 5>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 5>

<item 6>
Lake Osborne
Lake Osborne, Florida, USA is a 378-acre (152.9 hectares) lake that is part of a system of once natural freshwater lakes lying along the western slope of the
coastal ridge in Palm Beach County just west of the Florida Intracoastal Waterway and Atlantic Ocean.

 It is located within the C-16 drainage basin which occupies approximately 40,031 acres of land (16,200 hectares).

 Five drainage canals discharge directly to Lake Osborne, and Lake Osborne discharges via the E-4 canal to the C16 and C51 canals to the Lake Worth Lagoon.

 The lake is bordered on the west by John Prince Memorial Park, and on the east be the City of Lake Worth.
</item 6>

<item 7>
Lake Worth Lagoon
The Lake Worth Lagoon is a lagoon located in Palm Beach County, Florida.

 It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.

 The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
</item 7>

<item 8>
Tamiami Canal
The Tamiami Canal or C-4 Canal, is a canal located in southern Florida in the United States.

 It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center
near the Miami River.

 It averages 8 ft in depth and is over 100 ft wide in some areas.
</item 8>

<item 9>
Lake Worth, Florida
Lake Worth is a city in Palm Beach County, Florida, United States, which takes its name from the body of water along its eastern border, originally called "Lake
Worth", and now generally known as the Lake Worth Lagoon.

 The lake itself was named for General William J. Worth, who led U.S. forces during the last part of the Second Seminole War.

 s of 2010 , the population estimated by the U.S. Census Bureau was 34,910.

 The city is included in the Miami-Fort Lauderdale-Pompano Beach Metropolitan Area, which is home to approximately 5,563,857 people.
</item 9>

<item 10>
South Lake Worth Inlet
The South Lake Worth Inlet, also known as the Boynton Inlet, is an artificial cut through a barrier beach connecting the south end of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 The inlet is 130 ft wide and 6 ft and 12 ft deep.
</item 10>
=== END ITEM id=49956d68f0140e64 ===

=== ITEM id=a42d6d3f31ea57b3 ===
Gold sentences and facts:
Sentence 0: The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna (Tenerife, Canary Islands, Spain).
  Fact 0.0: The Church of the Guanche People's Spanish name is 'Iglesia del Pueblo Guanche'.
  Fact 0.1: The Church of the Guanche People is a religious organisation.
  Fact 0.2: The Church of the Guanche People was founded in 2001.
  Fact 0.3: The Church of the Guanche People was founded in the city of San Cristobal de La Laguna.
  Fact 0.4: San Cristobal de La Laguna is in Tenerife, Canary Islands, Spain.
Sentence 1:  Tenerife is the largest and most populous island of Macaronesia.
  Fact 1.0: Tenerife is the largest island of Macaronesia.
  Fact 1.1: Tenerife is the most populous island of Macaronesia.

Retrieved items:
<item 1>
Chiguata District
Chiguata is a district located in the Arequipa province, 30 km away from the city of Arequipa, Peru, on the slopes of the volcano Pikchu Pikchu (where at its
summit pre-Inca worshipped remains and artifacts have been found).

 The name "chiguata" comes from the Quechua words "chiri" meaning "cold" and "wata" meaning "year" (cold all year).

 Ancient Chiguata was inhabited by the local natives until the arrival of Spanish influence in charge of the emissary Diego Hernández de Córdova in 1540.

 Soon after, the construction of a small plaza was initiated, and according to the Dominican religious order and local sources, the plaza's "Templo del Espíritu
Santo" ("Sacred Spirit temple") along with its domes and lateral cover was not begun until 1739.

 The design of the temple is quite interesting, with two angels in feathered nightwear and their arms lifted vertically, separated by twelve fringes of
rectangular flowers which add to the whole decoration of the temple.
</item 1>

<item 2>
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
</item 2>

<item 3>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 3>

<item 4>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 4>

<item 5>
Guanche mummies
Guanche mummies are the intentionally desiccated remains of members of the indigenous Guanche people of the Canary Islands.

 The majority of Guanche mummies were made during the eras prior to Spanish settlement of the area in the 15th century.

 The methods of embalming are similar to those that were used by the Ancient Egyptians, though fewer mummies remain from the Guanche due to looting and
desecration.
</item 5>

<item 6>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 6>

<item 7>
Chiguata District
Cherubs also decorate the circular cornice, along with four saints in full-length reliefs that occupy the "pechinas" of the columns that sustain the temple's
dome, which is considered one of the most original works of the colony and now visited by occasional tourists.
</item 7>

<item 8>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 8>

<item 9>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 9>

<item 10>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 10>
=== END ITEM id=a42d6d3f31ea57b3 ===

=== ITEM id=5c8bbea7025c9c41 ===
Gold sentences and facts:
Sentence 0: In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
  Fact 0.0: In the middle of 1984, a Brazilian company called Prológica brought a new equipment to the Brazilian market
  Fact 0.1: Prológica made its own versions of 8 bits US computers
  Fact 0.2: The new equipment was for Prológica's personal computer series
  Fact 0.3: The new equipment was called "CP"
  Fact 0.4: CP is short for Personal Computer in Portuguese
Sentence 1: The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.
  Fact 1.0: The Atari ST is a line of home computers from Atari Corporation
  Fact 1.1: The Atari ST is the successor to the Atari 8-bit family
Sentence 2:  The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.
  Fact 2.0: The Atari ST is the first personal computer to come with a bitmapped color GUI
  Fact 2.1: The Atari ST's bitmapped color GUI uses a version of Digital Research's GEM
  Fact 2.2: The version of Digital Research's GEM used by the Atari ST was released in February 1985

Retrieved items:
<item 1>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 1>

<item 2>
Page 6
Page 6 (subtitled "Atari Users Magazine", and later known as New Atari User) was an independent British publication aimed at users of Atari home computers.

 It was published between 1982 and 1998.

 The computer magazine supported both the Atari 8-bit family of computers (400/800/XL/XE) and the Atari ST range.
</item 2>

<item 3>
Happy drives
Happy drives are series of disk drive enhancements for the Atari 8-bit and Atari ST computer families produced by a small company Happy Computers.

 Happy Computers is most noted for the add-in boards for the Atari 810 and Atari 1050 disk drives, which achieved a tremendous speed improvement for reading and
writing, and for the ability to "back up" floppies.

 Happy's products were among the most popular Atari computer add-ons.

 They were still in use and active in the aftermarket as of 2009.
</item 3>

<item 4>
XaAES
FreeMint, MultiTOS, and a few more obscure OSes are all successors to TOS the proprietary operating system of the Atari ST computer.

 Later models of the ST Computer were called the TT and the Falcon.

 TOS stands for The Operating System or Tramiel Operating System, as Jack Tramiel was the owner of Atari.

 GEMDOS itself was a near clone of DOS, the IBM PC OS.

 This was most visible to users in the 8.3 character file naming system.

 Like all personal computer operating systems of the day, TOS was single tasking.

 GEM, the graphical user interface, was licensed from Digital Research and was not included in Apple's lawsuit against DR, thus the Macintosh like appearance
and ways of doing things remained on Atari computers.

 As time went on, the big goal for the ST within and without Atari was to have a multitasking TOS.
</item 4>

<item 5>
ATASCII
The ATASCII character set, from ATARI Standard Code for Information Interchange", alternatively ATARI ASCII", is the variation on ASCII used in the Atari 8-bit
family of home computers.

 The first of this family were the Atari 400 and 800, released in 1979, and later models were released throughout the 1980s.

 The last computer to use the ATASCII character set was the Atari XEGS which was released in 1987.

 The Atari ST family of computers used the different Atari ST character set.
</item 5>

<item 6>
Megamax C
Megamax C is a K&R implementation of C for the Atari ST, Apple IIgs, and Macintosh computers.

 Sold by Megamax, Inc., based in Richardson, Texas, the package includes a one-pass compiler, linker, editor, resource construction kit, and documentation.

 In the early days of the Atari ST, Megamax C was the primary competitor to the Alcyon C compiler from Digital Research which was included in the official
developer kit from Atari.

 According to a review in "Antic" magazine, Megamax C compiled a small benchmark program six times faster than Digital Research's compiler.
</item 6>

<item 7>
DTACK Grounded
DTACK Grounded was a computer hobbyist newsletter published from July 1981 to September 1985 by Hal W. Hardenberg.

 Subtitled "The Journal of Simple 68000 Systems", the newsletter was dedicated to the proposition that the Motorola 68000 CPU could be used to build simple fast
computers.

 In 1981 this was a revolutionary idea.

 This was before 68000-based personal computers like the Apple Macintosh, Commodore Amiga, and Atari ST had been created.

 In 1981 Motorola was marketing the 68000 solely as a CPU for high-end Unix workstations.

 Hal believed that the 68000 could be used as a simple embedded microprocessor as well, and used the newsletter to explain how to do that.
</item 7>

<item 8>
Prológica CP-400
In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment
for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
</item 8>

<item 9>
Atari ST
The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.

 The initial ST model, the 520ST, saw limited release in April-June 1985 and was widely available in July.

 The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.

 The 1040ST, released in 1986, is the first personal computer to ship with a megabyte of RAM in the base configuration and also the first with a
cost-per-kilobyte of less than US$1.
</item 9>

<item 10>
Atari Corporation
Atari Corporation was an American manufacturer of computers and video game consoles from 1984 to 1996.

 Atari Corp. was founded in July 1984 when Warner Communications sold the home computing and game console divisions of Atari, Inc. to Jack Tramiel.

 Its chief products were the Atari ST, , Atari 7800, Atari Lynx, and Atari Jaguar.

 The company reverse merged with JTS Inc. in 1996, becoming a small division, which itself closed when JTS liquidated the IP to Hasbro Interactive in 1998.
</item 10>
=== END ITEM id=5c8bbea7025c9c41 ===

=== ITEM id=c237ef1f36f27cf1 ===
Gold sentences and facts:
Sentence 0: Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
  Fact 0.0: Tímea Babos was born on 10 May 1993.
  Fact 0.1: Tímea Babos is a Hungarian tennis player.
Sentence 1: Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis Professionals (ATP).
  Fact 1.0: Jack Sock was born on September 24, 1992.
  Fact 1.1: Jack Sock is an American professional tennis player.
  Fact 1.2: Jack Sock is currently ranked world No. 21 in men's singles.
  Fact 1.3: The ranking is by the Association of Tennis Professionals (ATP).

Retrieved items:
<item 1>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 1>

<item 2>
2016 Dubai Tennis Championships – Women's Doubles
Tímea Babos and Kristina Mladenovic were the defending champions, but chose not to compete together.

 Babos played alongside Julia Görges, but lost in the semifinals to Caroline Garcia and Mladenovic.

<br>
</item 2>

<item 3>
Tímea Babos
Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
</item 3>

<item 4>
2013 Aegon Classic – Doubles
Tímea Babos and Hsieh Su-wei were the defending champions, but Babos decided not to participate.

<br>
</item 4>

<item 5>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 5>

<item 6>
2014 Suzhou Ladies Open – Doubles
Tímea Babos and Michaëlla Krajicek were the defending champions, having won the event in 2013, however both players chose not to participate.
</item 6>

<item 7>
2017 Apia International Sydney – Women's Doubles
Martina Hingis and Sania Mirza were the defending champions, but chose not to participate together this year.

 Hingis played alongside Coco Vandeweghe, but lost in the quarterfinals to Tímea Babos and Anastasia Pavlyuchenkova.

 Mirza teamed up with Barbora Strýcová, but lost in the final to Babos and Pavlyuchenkova, 4–6, 4–6.
</item 7>

<item 8>
2015 Malaysian Open – Doubles
Tímea Babos and Chan Hao-ching were the defending champions, but Chan chose not to participate this year.

 Babos chose to play in Monterrey, but lost in the first round.
</item 8>

<item 9>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 9>

<item 10>
Jack Sock
Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis
Professionals (ATP).

 He is one of the top ranked American men in both singles and doubles on the ATP Tour.

 A former junior US Open champion, Sock's singles success is highlighted by 7 ATP finals, including three titles.

 His career best result came at the 2017 Indian Wells Masters, where he reached the semi-finals.
</item 10>
=== END ITEM id=c237ef1f36f27cf1 ===

=== ITEM id=e1153a4cb4c6b65b ===
Gold sentences and facts:
Sentence 0: Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.
  Fact 0.0: Paul Clyne was the District Attorney of Albany County, New York.
  Fact 0.1: Paul Clyne held this office from January 2001.
  Fact 0.2: Paul Clyne held this office through December 2004.
Sentence 1: P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.
  Fact 1.0: P. David Soares was born on October 26, 1969.
  Fact 1.1: P. David Soares was born in Brava, Cape Verde.
  Fact 1.2: P. David Soares is the District Attorney of Albany County, New York.

Retrieved items:
<item 1>
Paul Clyne
Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.

 A graduate of Albany Law School, he spent about 14 years as an assistant district attorney, before he was tapped by local politicians to replace the retiring
District Attorney, Sol Greenberg.

 He was defeated for re-election by David Soares, first in the Democratic Party primary election in September 2004, and then in the general election in November
2004, in which he ran on an independent line.

 After a stint teaching at the New York Prosecutors Institute, he went into private practice as a criminal defense attorney in 2007, with an office in Albany,
New York.
</item 1>

<item 2>
Richard Pleasance
Richard Arnold Pleasance is an Australian rock musician and producer.

 He was a founding member of Boom Crash Opera on guitar, bass guitar, vocals and as a songwriter in 1985; they released three albums before Pleasance left in
1992.

 Their hit Australian Recording Industry Association (ARIA) singles, "Great Wall" (No. 5, 1986) and "Onion Skin" (No. 11, 1989) were co-written by Pleasance,
who also co-produced their second album, "These Here Are Crazy Times" (No. 10, 1989).

 His debut solo release, "Galleon" received four nominations at the ARIA Music Awards for 1992.

 Pleasance composed the theme music for Australian television series, "SeaChange" (1998–2001), for 2006 feature film "Kenny" and more recently he composed the
theme music for the prison drama series "Wentworth".

 Pleasance is married to Michelle and, as from May 2009, he was living in Hepburn Springs, Victoria where he has a recording studio.
</item 2>

<item 3>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 3>

<item 4>
City Cruises
City Cruises is a public limited company that operates scheduled public sightseeing cruises on the River Thames in London, serving Westminster, London Eye,
Tower and Greenwich piers every day throughout the year.

 It operates the services mainly using purpose-built wheelchair-accessible 'RiverLiner' vessels, which have a large open top deck and enclosed lower saloon with
panoramic windows and bars, backed up by some ex-Paris single-deck conservatory-style sightseeing boats.

 The company also owns a number of other boats with which it provides a variety of private charter and waterage services on the River Thames.

 City Cruises owns the London Showboat brand, under which it operates an evening dinner, cabaret and dance cruise from Westminster Pier throughout the year, and
also owns and operates the Restaurant Ship Hispaniola, which is permanently moored alongside Victoria Embankment close to Embankment Underground station in
central London.
</item 4>

<item 5>
Leah Flanagan is a Sydney-based Indigenous singer-songwriter who has released two albums and toured Australia; her latest, *Saudades*, was recorded at Jim
Moginie’s studio and mixed by Paul McKercher. Donegal’s Leo’s Tavern, opened in 1968 by Leo Brennan, launched the musical careers of his children, including
Moya, Enya and Clannad. Moya Brennan, Irish folk singer, harpist and Clannad member, has Grammy
</item 5>

<item 6>
The entries profile artists, performers, filmmakers, musicians, writers and cultural topics. AES+F is a Russian art collective known for monumental video
installations; Lev Yilmaz created *Tales of Mere Existence*, while Clinton Walker chronicles Australian culture. Other figures include South African actor Milan
Murray; Swedish entertainer Henrik Dorsin; Japanese actor Bokuzen Hidari; Turkish electronic composer Erdem Helvacıoğlu; Lynch collaborator Mary Sweeney;
Icelandic actor Balt
</item 6>

<item 7>
Paul Soniat
Paul Soniat is the Director of the New Orleans City Park Botanical Gardens.

 He was born at the old Touro Hospital in New Orleans.

 His family has been in New Orleans since 1727 and he grew up and later lived on the street bearing his family name.

 Paul is a self-taught piano player, and has released two CDs, absorbing the sights, sounds, and flavor of New Orleans.

 His first CD, titled "Born in New Orleans", was released in April 2005.

 His second CD was released after Hurricane Katrina hit the Gulf coast and the New Orleans' levees failed.

 This second CD is appropriately titled "Below the Water Line".
</item 7>

<item 8>
City Cruises runs year-round, wheelchair-accessible Thames sightseeing services between major London piers, plus private charters, the London Showboat dinner
cruise and permanently moored Hispaniola restaurant. The remaining entries describe London locations and institutions: Doughty Street Chambers’ legal work and
Doughty Street’s history; Westminster’s landmarks and former parishes; Abbey Road and its Beatles association; Westminster Underground station; Temple Bar’s
ceremonial role and Wren gateway; and the former Westminster Magistrates’
</item 8>

<item 9>
Skip Bolen
Skip Bolen is a Southern photographer of musicians, architecture, lifestyle and the culture of New Orleans.

 Born in Lafayette, Louisiana, he moved to New Orleans where he began his publishing career as a designer and art director.

 After moving to New York City, he began working at "House & Garden," renamed "HG," as Senior Designer in January 1988 with Anna Wintour and Alexander Liberman
at Condé Nast Publications.

 Spending evenings in jazz clubs, he began photographing jazz musicians in New York City and often when he regularly returned to New Orleans.

 After three years at Condé Nast Publications in New York City, he returned to New Orleans to pursue his jazz photography full-time.

 In 1998, he moved to Los Angeles where he became art director of House of Blues for seven years while photographing at night and weekends.

 He continued photographing jazz musicians and had his first major solo exhibition at the Jazz Bakery in Los Angeles on August 9, 2002.

 On July 4, 2006, he returned to New Orleans to pursue photography full-time documenting the recovery and rebuilding of New Orleans since Hurricane Katrina,
documenting the jazz scene, night-time photography and other photographic projects.
</item 9>

<item 10>
David Soares
P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.

 He is a Democrat.
</item 10>
=== END ITEM id=e1153a4cb4c6b65b ===

=== ITEM id=8ab76270dda703a9 ===
Gold sentences and facts:
Sentence 0: Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.
  Fact 0.0: Eriogonum is the scientific name for a genus of flowering plants.
  Fact 0.1: The genus Eriogonum is in the family Polygonaceae.
Sentence 1: Aloinopsis is a genus of ice plants from South Africa.
  Fact 1.0: Aloinopsis is a genus of ice plants.
  Fact 1.1: The Aloinopsis ice plants come from South Africa.

Retrieved items:
<item 1>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 1>

<item 2>
Eriogonum heracleoides
Eriogonum heracleoides (common names parsnipflower buckwheat, whorled buckwheat, and Wyeth buckwheat) has many flowering clusters that are cream or off-white in
color.

 Its usual habitat is rocky areas such as sagebrush deserts and Ponderosa pine forests.

 Parsnipflower buckwheat is in the genus "Eriogonum" and the family Polygonaceae which is a family of plants known as the "knotweed family".

 It inhabits much of the western part of the United States.
</item 2>

<item 3>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 3>

<item 4>
Eriogonum diclinum
Eriogonum diclinum is a species of wild buckwheat known by the common name Jaynes Canyon buckwheat.

 It is native to the Klamath Mountains of northern California and southern Oregon, where it is an uncommon member of plant communities on serpentine soils.

 This is a small dioecious shrub forming low, thick mats rarely exceeding 20 centimeters in height.

 The curving, oval-shaped leaves are woolly and greenish gray in color, growing up to two centimeters long.

 The plant flowers in yellow to reddish rounded inflorescences, male plants producing clusters of staminate flowers, and female plants producing larger clusters
of pistillate flowers.
</item 4>

<item 5>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 5>

<item 6>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 6>

<item 7>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 7>

<item 8>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 8>

<item 9>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 9>

<item 10>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 10>
=== END ITEM id=8ab76270dda703a9 ===

=== ITEM id=6247ac6580b2f67e ===
Gold sentences and facts:
Sentence 0: Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
  Fact 0.0: Mauritz Stiller was born on 17 July 1883.
  Fact 0.1: Mauritz Stiller died on 18 November 1928.
  Fact 0.2: Mauritz Stiller was a Finnish-Swedish film director.
  Fact 0.3: Mauritz Stiller was best known for discovering Greta Garbo.
  Fact 0.4: Mauritz Stiller brought Greta Garbo to America.
Sentence 1: Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.
  Fact 1.0: Johannes Grenzfurthner was born in 1975 in Vienna.
  Fact 1.1: Johannes Grenzfurthner is an award-winning Austrian artist.
  Fact 1.2: Johannes Grenzfurthner is an Austrian filmmaker.
  Fact 1.3: Johannes Grenzfurthner is an Austrian writer.
  Fact 1.4: Johannes Grenzfurthner is an Austrian actor.
  Fact 1.5: Johannes Grenzfurthner is an Austrian curator.
  Fact 1.6: Johannes Grenzfurthner is an Austrian theatre director.
  Fact 1.7: Johannes Grenzfurthner is an Austrian lecturer.

Retrieved items:
<item 1>
Graph query row:
person: Johannes Grenzfurthner
nationality: Austrian
</item 1>

<item 2>
Mauritz Stiller
Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
</item 2>

<item 3>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 3>

<item 4>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 4>

<item 5>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 5>

<item 6>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 6>

<item 7>
Traceroute (film)
Traceroute is a 2016 Austrian/American documentary film directed by Johannes Grenzfurthner.

 The autobiographical documentary and road movie deals with the history, politics and impact of nerd culture.

 Grenzfurthner calls his film a "personal journey into the uncharted depths of nerd culture, a realm full of dangers, creatures and more or less precarious
working conditions", an attempt to "chase the ghosts of nerddom's past, present and future."

 The film was co-produced by art group monochrom and Reisenbauer Film.

 It features music by Kasson Crooker, Hans Nieswandt, and many others.
</item 7>

<item 8>
Johannes Grenzfurthner
Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.

 He is known as the founder, conceptualist and artistic director of monochrom, an international art and theory group.

 Most of his artworks are labelled "monochrom".
</item 8>

<item 9>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 9>

<item 10>
Graph query row:
person: Mauritz Stiller
nationality: Finnish-Swedish
</item 10>
=== END ITEM id=6247ac6580b2f67e ===

=== ITEM id=7f0ea11af38839fc ===
Gold sentences and facts:
Sentence 0: Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.
  Fact 0.0: Emmitt H. Ford was born on December 13, 1943.
  Fact 0.1: Emmitt H. Ford died on November 10, 2014.
  Fact 0.2: Emmitt H. Ford was an American politician.
  Fact 0.3: Emmitt H. Ford was from Tennessee.
Sentence 1:  He was also an uncle of former United States Congressman Harold Ford Jr.
  Fact 1.0: Emmitt H. Ford was an uncle of Harold Ford Jr.
  Fact 1.1: Harold Ford Jr. is a former United States Congressman.
Sentence 2:  He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.
  Fact 2.0: Emmitt H. Ford is the son of Harold Ford Sr.
  Fact 2.1: Harold Ford Sr. is a former Congressman.
  Fact 2.2: Harold Ford Sr. held the same seat for 22 years.

Retrieved items:
<item 1>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 1>

<item 2>
United States Senate election in Tennessee, 2006
The 2006 United States Senate election in Tennessee was held on November 7, 2006.

 The election winner, Bob Corker, served his term between January 3, 2007 and January 3, 2013.

 Corker replaced Republican Bill Frist in the Senate, who retired upon the end of his second term in 2007.

 Corker was the Republican nominee, and the Democratic nominee was Harold Ford, Jr..

 The race between Ford and Corker was one of the most competitive Senate races of 2006, with Corker winning the race by less than three percent of the vote.

 Corker was the only non-incumbent Republican to win a U.S. Senate seat in 2006.

 Since 1994, the Republican Party has held both of Tennessee's Senate seats.
</item 2>

<item 3>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 3>

<item 4>
Doug Ford Jr.
Douglas Bruce Ford Jr. (born November 20, 1964) is a Canadian businessman and politician in Toronto, Ontario.

 Ford was Toronto City Councillor for Ward 2 Etobicoke North in Toronto from 2010 to 2014 at the same time that his brother, Rob Ford (1969–2016), was mayor of
Toronto.

 Their father, Doug Ford Sr., was a Member of Provincial Parliament from 1995 to 1999 and founded Deco Labels and Tags, a printing business operating in Canada
and the United States.

 The business is now owned by Doug Ford Jr., who has served as company president since 2002, and his brother, Randy.
</item 4>

<item 5>
Robert Philibosian
Robert Harry Philibosian (born 1940) is an American politician.

 He served as Los Angeles County District Attorney from 1982 to 1984.

 He received his B.A. in history from Stanford University and his law degree from Southwestern Law School, and was admitted to the California State Bar in 1968.

 He is now Of counsel at the law firm of Sheppard, Mullin, Richter & Hampton.
</item 5>

<item 6>
L. R. Ford Jr.
Lester Randolph Ford Jr. (born September 23, 1927 – February 26, 2017) was an American mathematician specializing in network flow problems.

 He was the son of mathematician Lester R. Ford Sr.
</item 6>

<item 7>
George McKelvey (mayor)
George M. McKelvey is an American politician.

 A Democrat, he served two terms as Mayor of Youngstown, Ohio, from 1998 to 2005.

 Prior to serving as mayor, he served two terms as the Treasurer of Mahoning County.

 He has also been a teacher, school administrator and city council member in Youngstown.
</item 7>

<item 8>
Harold Ford Jr.
Harold Eugene Ford Jr. (born May 11, 1970) is an American politician and author who served five terms in the United States House of Representatives as a member
of the Democratic Party from  's 9 congressional district , centered in Memphis, from 1997 to 2007.

 He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.

 In 2006, Ford made an unsuccessful bid for the US Senate seat vacated by retiring Bill Frist.

 He is a member of the Ford political family from Memphis.

 Ford was the last chairman of the Democratic Leadership Council (DLC).
</item 8>

<item 9>
Emmitt Ford
Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.

 He represented the 86th district encompassing Shelby County from 1975–1981.

 He was also an uncle of former United States Congressman Harold Ford Jr.
</item 9>

<item 10>
John Ford (Tennessee politician)
John N. Ford (born May 3, 1942 in Memphis, Tennessee), is a former Democratic member of the Tennessee State Senate and a member of Tennessee's most prominent
African-American political family.

 He is the older brother of former U.S. Representative Harold Ford, Sr. and the uncle of former Tennessee U.S. Representative and 2006 United States Senate
candidate Harold Ford, Jr. He is a member of the Ford political family from Memphis.
</item 10>
=== END ITEM id=7f0ea11af38839fc ===

=== ITEM id=2e2351595e177b69 ===
Gold sentences and facts:
Sentence 0: The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.
  Fact 0.0: The Shins are an American indie rock band.
  Fact 0.1: The Shins are from Albuquerque, New Mexico.
  Fact 0.2: The Shins were formed in 1996.
Sentence 1: Paramore is an American rock band from Franklin, Tennessee, formed in 2004.
  Fact 1.0: Paramore is an American rock band.
  Fact 1.1: Paramore is from Franklin, Tennessee.
  Fact 1.2: Paramore was formed in 2004.

Retrieved items:
<item 1>
Singles Club (EP)
Singles Club is an EP released by American rock band Paramore.

 The songs were released as promotional singles between October and December 2011, culminating in the release of a box set containing the three constituent
songs plus "Monster", which was recorded during the same sessions.

 The EP and box set were released on December 14, 2011 on Paramore's website.

 The EP is the first release of new material by Paramore not to include band members Josh and Zac Farro.
</item 1>

<item 2>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 2>

<item 3>
Paramore (album)
Paramore is the self-titled fourth studio album by American rock band Paramore.

 It was released on April 5, 2013, through Fueled by Ramen as a follow-up to "Brand New Eyes" (2009).

 It is their first full-length album released after the departure of co-founders Josh and Zac Farro in 2010.

 Recorded between April and November 1, 2012, the album was described by the band as being a "statement" and a reintroduction of the band to the world and to
themselves.

 It is the only album without Zac Farro on drums, and the final album recorded with bassist Jeremy Davis before he left the band in 2015.
</item 3>

<item 4>
After Laughter
After Laughter is the fifth studio album by American rock band Paramore.

 It was released on May 12, 2017, through Fueled by Ramen as a follow-up to "Paramore", their 2013 self-titled album.

 The album was produced by guitarist Taylor York alongside previous collaborator, Justin Meldal-Johnsen.

 It is the band's first album since the return of drummer Zac Farro, who left the band with his brother Josh in 2010, and the departure of former bassist Jeremy
Davis, who left the band in 2015.

 "After Laughter" represents a complete departure from the usual pop punk and alternative rock sound of their previous releases.

 The album touches on themes of exhaustion, depression and anxiety, contrasting the upbeat and vibrant sound of the record.
</item 4>

<item 5>
List of songs recorded by Paramore
American rock band Paramore have recorded songs for five studio albums, a box set, an extended play and two soundtrack albums.

 In 2002, at age 13, vocalist Hayley Williams moved to Franklin, Tennessee, where she met brothers Josh Farro and Zac Farro.

 The band was officially formed by Josh Farro (lead guitar and backing vocals), Zac Farro (drums), Jeremy Davis (bass guitar) and Williams (lead vocals) in
2004, with the later addition of Williams' neighbor Jason Bynum (rhythm guitar).

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling" that year.

 Three singles were released to promote the album.
</item 5>

<item 6>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 6>

<item 7>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 7>

<item 8>
Paramore discography
The American rock band Paramore has released five studio albums, three extended plays, two live albums, eighteen singles, one video album, and nineteen music
videos.

 The band was formed in Franklin, Tennessee, in 2004 by lead vocalist Hayley Williams with guitarists Josh Farro and Taylor York, bassist Jeremy Davis, and
drummer Zac Farro.

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling".

 Three singles were released from the album, but none of them charted.

 The album did not chart in the "Billboard" 200 either, although it peaked at number thirty in the "Billboard" Top Heatseekers.

 "All We Know Is Falling" received Gold certification in the United Kingdom and in July 2014 the RIAA certified the album Gold in the United States.
</item 8>

<item 9>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 9>

<item 10>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 10>
=== END ITEM id=2e2351595e177b69 ===

=== ITEM id=3a99444ee5b42be7 ===
Gold sentences and facts:
Sentence 0:  Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest commercial success is the erotic thriller "Chloe" (2009).
  Fact 0.0: Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter".
  Fact 0.1: "The Sweet Hereafter" was released in 1997.
  Fact 0.2: Egoyan received two Academy Award nominations for "The Sweet Hereafter".
  Fact 0.3: Egoyan's biggest commercial success is the erotic thriller "Chloe".
  Fact 0.4: "Chloe" was released in 2009.
Sentence 1: Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".
  Fact 1.0: Chloe is a 2009 erotic thriller film.
  Fact 1.1: Chloe was directed by Atom Egoyan.
  Fact 1.2: Chloe is a remake of the 2003 French film "Nathalie...".

Retrieved items:
<item 1>
Beyond the Sea (1991 film)
Beyond the Sea (translit.

 Me'ever Layam) is a 1991 Israeli drama film directed by Jacob Goldwasser.

 It stars Aryeh Moskona, Dafna Rechter, Moti Giladi.

 The film, a family drama set in the 1960s, was critically acclaimed, and won the Ophir Award for Best Film.

 The film was selected as the Israeli entry for the Best Foreign Language Film at the 64th Academy Awards, but was not accepted as a nominee.
</item 1>

<item 2>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 2>

<item 3>
Tom Cruise filmography
His next role was as anti-war activist Ron Kovic in the drama adaptation of Kovic's memoir of the same name, "Born on the Fourth of July" (1989).

 For his performance Cruise received the Golden Globe Award for Best Actor – Motion Picture Drama and his first nomination for the Academy Award for Best Actor.
</item 3>

<item 4>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 4>

<item 5>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
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
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 7>

<item 8>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 8>

<item 9>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 9>

<item 10>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 10>
=== END ITEM id=3a99444ee5b42be7 ===

=== ITEM id=5c80d722867ae89a ===
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
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 1>

<item 2>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 2>

<item 3>
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 3>

<item 4>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 4>

<item 5>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 5>

<item 6>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 6>

<item 7>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 7>

<item 8>
Yorktown Heights (album)
Yorktown Heights was the only solo album by Feeder frontman Grant Nicholas.

 The album was released on 11 August 2014 in the United Kingdom and recorded during 2013, being a year after Nicholas announced at Feeder's final show of 2012
at the Brixton Academy that the band would be taking a break.

 It was with the first airplay of ""Soul Mates"" on the BBC Radio 2 show "The Dermot O'Leary Show" on 31 May 2014, that Grant's solo career made its radio
debut.
</item 8>

<item 9>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 9>

<item 10>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 10>
=== END ITEM id=5c80d722867ae89a ===

=== ITEM id=3ad7dbe7b148bc4c ===
Gold sentences and facts:
Sentence 0: Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.
  Fact 0.0: Neon City is also known as Anno 2053 in Italy.
  Fact 0.1: Neon City is also known as Neonski Grad in Serbia.
  Fact 0.2: Neon City is a 1991 Canadian film.
  Fact 0.3: Neon City is a post-apocalyptic science fiction film.
  Fact 0.4: Neon City was directed by Monte Markham.
  Fact 0.5: Neon City was written by Jeff Begun.
  Fact 0.6: Neon City was written by Ann Lewis Hamilton.
  Fact 0.7: Neon City's writing was credited under the pseudonym Buck Finch.
Sentence 1:  The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
  Fact 1.0: Michael Ironside is in the cast of Neon City.
  Fact 1.1: Vanity is in the cast of Neon City.
  Fact 1.2: Lyle Alzado is in the cast of Neon City.
  Fact 1.3: Valerie Wildman is in the cast of Neon City.
  Fact 1.4: Nick Klar is in the cast of Neon City.
  Fact 1.5: Juliet Landau is in the cast of Neon City.
  Fact 1.6: Richard Sanders is in the cast of Neon City.
  Fact 1.7: Monte Markham is in the cast of Neon City.
Sentence 2: Lyle Martin Alzado (April 3, 1949 – May 14, 1992) was a professional American football defensive end of the National Football League famous for his intense and intimidating style of play.
  Fact 2.0: Lyle Martin Alzado was born on April 3, 1949.
  Fact 2.1: Lyle Martin Alzado died on May 14, 1992.
  Fact 2.2: Lyle Martin Alzado was a professional American football defensive end.
  Fact 2.3: Lyle Martin Alzado played in the National Football League.
  Fact 2.4: Lyle Martin Alzado was famous for his intense and intimidating style of play.

Retrieved items:
<item 1>
The entries cover diverse films and careers: crime drama *Deadfall* and its *Arsenal* follow-up; *Rogue One*’s rebels stealing Death Star plans; Mel Gibson,
Nicolas Cage, James Garner and Tom Cruise filmographies; Everest disaster drama *Everest*; integration-themed sports films *Remember the Titans* and *McFarland,
USA*; acclaimed *True Grit* and *The Hurt Locker*; espionage film *The Ipcress File*; Vietnam
</item 1>

<item 2>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 2>

<item 3>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 3>

<item 4>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 4>

<item 5>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 5>

<item 6>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 6>

<item 7>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 7>

<item 8>
The collection spans films and TV from 1947–2010: Bruce Willis’s dramas, action comedies and thrillers include *In Country* (Golden Globe-nominated), *Last Man
Standing* (a *Yojimbo* remake), *The Whole Nine Yards* and *Cop Out*. Other highlights: *The Bears and I*; teen comedies *Can’t Buy Me Love* and *The Night
Before*; *The Country Girl* (Oscar-winning screenplay
</item 8>

<item 9>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 9>

<item 10>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 10>
=== END ITEM id=3ad7dbe7b148bc4c ===
