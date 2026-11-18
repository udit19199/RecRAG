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

=== ITEM id=00932db77b9adbc2 ===
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
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 1>

<item 2>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 2>

<item 3>
Basset Bleu de Gascogne
The Basset Bleu de Gascogne (] ), also known as the Blue Gascony Basset, is a long-backed, short legged breed of dog of the hound type.

 The breed originated in the Middle Ages, descended from the Grand Bleu de Gascogne.

 It nearly became extinct around the early 19th century; its salvation was attributed to one Alain Bourbon.

 A French native breed, it is rare outside of its homeland.

 It is recognized internationally by the Fédération Cynologique Internationale, in the UK by The Kennel Club, and by the United Kennel Club in the United
States.

 The "bleu" of its name is a reference to its coat which has a ticked appearance.
</item 3>

<item 4>
Bulldog
The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.

 Other scent-hound breeds include the Small Greek Domestic Dog, Irish Wolfhound, Bluetick Coonhound, Finnish Lapphund, and the Basset Hound.

 The Bulldog is a muscular, hefty dog with a wrinkled face and a distinctive pushed-in nose.

 The American Kennel Club (AKC), The Kennel Club (UK), and the United Kennel Club (UKC) oversee breeding records.

 Bulldogs were the fourth most popular purebreed in the US in 2007 according to the American Kennel Club.
</item 4>

<item 5>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 5>

<item 6>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 6>

<item 7>
Ariegeois
The Ariegeois is a breed of dog from the "département" of Ariège in the Midi-Pyrenées region of southern France.

 It is a medium-sized pack-hunting scenthound deriving from crossing of Grand Bleu de Gascogne and Grand Gascon-Saintongeois hounds with local Briquet dogs.

 It is used both as a courser and for driving game to waiting guns.

 While most successful with hares, it is also used for hunting deer and boar.

 It is distinguished by its friendly nature with other hounds and affection for human companions.
</item 7>

<item 8>
Petit Bleu de Gascogne
The Petit Bleu de Gascogne (FCI No.31) Is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog.

 The Petit Bleu de Gascogne is not a small (petite) dog, the name comes from its use on small game.
</item 8>

<item 9>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 9>

<item 10>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 10>
=== END ITEM id=00932db77b9adbc2 ===

=== ITEM id=3d1458d581ec0177 ===
Gold sentences and facts:
Sentence 0: Wheelock Whitney may refer to one of three members of the Whitney family:
  Fact 0.0: Wheelock Whitney may refer to one of three members of the Whitney family.
Sentence 1: The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673) who came from London, England to Watertown, Massachusetts in 1635.
  Fact 1.0: The Whitney family is an American family.
  Fact 1.1: The Whitney family is notable for their social prominence, wealth, business enterprises and philanthropy.
  Fact 1.2: The Whitney family was founded by John Whitney (1592–1673).
  Fact 1.3: John Whitney came from London, England.
  Fact 1.4: John Whitney came to Watertown, Massachusetts.
  Fact 1.5: John Whitney came to Watertown, Massachusetts in 1635.

Retrieved items:
<item 1>
James Scollay Whitney
James Scollay Whitney (May 19, 1811 – October 24, 1878) was an American business executive and politician.

 He was the father of Henry Melville Whitney and William Collins Whitney, founders of the Whitney family business interests.
</item 1>

<item 2>
Ross Winans
Ross Winans (1796–1877) was an American inventor, mechanic, and builder of locomotives and railroad machinery.

 He is also noted for design of pioneering cigar-hulled ships.

 Winans, one of the United States' first multi-millionaires, was involved in national and state politics, a southern-sympathizer and was a vehement "states'
rights" advocate.

 His outspoken anti-federal stance as a member of the Maryland House of Delegates, the lower chamber of the General Assembly, (state legislature) led to his
temporary arrest on board a Baltimore and Ohio Railroad train returning from an early session of the legislature held in the western Maryland town of Frederick
to avoid the Union Army-occupied state capital of Annapolis in April–May, 1861, to consider the possibilities of state secession, during the early decisive
period of the American Civil War.

 Winans was related to James McNeill Whistler through marriage (Whistler's brother George married Winans' daughter Julia).
</item 2>

<item 3>
Harry Payne Whitney
Harry Payne Whitney (April 29, 1872 – October 26, 1930) was an American businessman, thoroughbred horse breeder, and member of the prominent Whitney family.
</item 3>

<item 4>
Charles Andrew Whitney
Charles Andrew Whitney (November 14, 1864- December 31, 1912) was an American businessman and industrialist in the late 19th century, born in Princeton,
Massachusetts.

 He was part of the prominent American Whitney family.

 In 1859, Charles, his brother Levi L. Whitney, and Orville E. Thompson helped lead the large-scale manufacturing of leather boots and shoes in Chicago and were
attributed with successfully running the first factory of its kind there.

 Whitney himself held patents for the manufacture of leather.
</item 4>

<item 5>
John Hay Whitney
John Hay Whitney (August 17, 1904 – February 8, 1982), colloquially known as Jock Whitney, was U.S. Ambassador to the United Kingdom, publisher of the "New York
Herald Tribune", and president of the Museum of Modern Art.

 He was a member of the Whitney family.
</item 5>

<item 6>
Gertrude Vanderbilt Whitney
Gertrude Vanderbilt Whitney (January 9, 1875 – April 18, 1942) was an American sculptor, art patron and collector, and founder in 1931 of the Whitney Museum of
American Art in New York City.

 She was a prominent social figure and hostess, who was born into the wealthy Vanderbilt family and married into the Whitney family.
</item 6>

<item 7>
Greentree Stable
Greentree Stable, in Red Bank, New Jersey, was a major American thoroughbred horse racing stable and breeding farm established in 1914 by Payne Whitney of the
Whitney family of New York City.

 Payne Whitney operated a horse farm and stable at Saratoga Springs, New York with his brother Harry Payne Whitney, who also had a large stable of horses.

 Greentree Stable had a training base at Aiken, South Carolina, while Greentree Farm in Lexington, Kentucky was established in 1925 as its breeding arm.
</item 7>

<item 8>
Wheelock Whitney
Wheelock Whitney may refer to one of three members of the Whitney family:
</item 8>

<item 9>
Biddle family
The American Biddle family of Philadelphia began when William Biddle (1630–1712) and Sarah Kempe (1634–1709), Quakers, emigrated from England to America in
1681, in part, to avoid religious persecution.

 Having acquired extensive rights to over 43000 acre of lands in Quaker West Jersey, they settled first at Burlington.

 Two third generation brothers, William Biddle, 3rd (1698–1756), and John Biddle (1707–1789), moved from Mount Hope (1684) near Bordentown to Philadelphia in
the 1720s and 1730s and constituted the first generation of the Philadelphia Biddle family.
</item 9>

<item 10>
Whitney family
The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673)
who came from London, England to Watertown, Massachusetts in 1635.

 The historic family mansion in Watertown, known as The Elms, was built in 1710.
</item 10>
=== END ITEM id=3d1458d581ec0177 ===

=== ITEM id=8975a66e15f0c927 ===
Gold sentences and facts:
Sentence 0: Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television host/personality who "TV Guide" called "The Charlie Chaplin of Television".
  Fact 0.0: Alan Young was born Angus Young.
  Fact 0.1: Alan Young was born on November 19, 1919.
  Fact 0.2: Alan Young died on May 19, 2016.
  Fact 0.3: Alan Young was a British-born Canadian-American actor.
  Fact 0.4: Alan Young was a voice actor.
  Fact 0.5: Alan Young was a comedian.
  Fact 0.6: Alan Young was a radio and television host/personality.
  Fact 0.7: TV Guide called Alan Young "The Charlie Chaplin of Television".
Sentence 1:  Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983) and in various other films, TV series and video games until his death.
  Fact 1.0: Alan Young was the voice of Disney's Scrooge McDuck for over thirty years.
  Fact 1.1: Alan Young first voiced Scrooge McDuck in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983).
  Fact 1.2: Alan Young voiced Scrooge McDuck in various other films, TV series and video games.
  Fact 1.3: Alan Young voiced Scrooge McDuck until his death.
Sentence 2:  During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy Awards in 1951.
  Fact 2.0: During the 1940s and 1950s, Alan Young starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television.
  Fact 2.1: The television version of "The Alan Young Show" gained Alan Young two Emmy Awards in 1951.
Sentence 3:  The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.
  Fact 3.0: The cartoon is an adaptation of Charles Dickens' "A Christmas Carol".
  Fact 3.1: The cartoon stars Scrooge McDuck as Ebenezer Scrooge.

Retrieved items:
<item 1>
Scrooge (1913 film)
Scrooge is a 1913 British black and white silent film based on the 1843 novel "A Christmas Carol" by Charles Dickens.

 It starred Seymour Hicks as Ebenezer Scrooge.

 In the United States it was released in 1926 as "Old Scrooge".
</item 1>

<item 2>
A Christmas Carol
A Christmas Carol in Prose, Being a Ghost-Story of Christmas, commonly known as A Christmas Carol, is a novella by Charles Dickens, first published in London by
Chapman & Hall in 1843; the first edition was illustrated by John Leech.

 "A Christmas Carol" tells the story of Ebenezer Scrooge, an old miser who is visited by the ghost of his former business partner Jacob Marley and the Ghosts of
Christmas Past, Present and Yet to Come.

 After their visits Scrooge is transformed into a kinder, gentler man.
</item 2>

<item 3>
A Christmas Carol (2009 film)
A Christmas Carol is a 2009 American 3D computer animated motion-capture fantasy film written and directed by Robert Zemeckis.

 It is an adaptation of the Charles Dickens story of the same name and stars Jim Carrey in a multitude of roles, including Ebenezer Scrooge as a young,
middle-aged, and old man, and the three ghosts who haunt Scrooge.

 The film also features supporting roles done by Gary Oldman, Colin Firth, Bob Hoskins, Robin Wright, and Cary Elwes.
</item 3>

<item 4>
Scrooge (1951 film)
Scrooge is a 1951 British fantasy drama film and an adaptation of Charles Dickens' "A Christmas Carol" (1843).

 It stars Alastair Sim as Ebenezer Scrooge, and was produced and directed by Brian Desmond Hurst, with a screenplay by Noel Langley.

 It was released as A Christmas Carol in the United States.
</item 4>

<item 5>
Scrooge (1935 film)
Scrooge is a 1935 British fantasy film directed by Henry Edwards and starring Seymour Hicks, Donald Calthrop and Robert Cochran.

 Hicks appears as Ebenezer Scrooge, the miser who hates Christmas.

 It was the first sound version of the Charles Dickens classic "A Christmas Carol", not counting a 1928 short subject that now appears to be lost.

 Hicks had previously played the role of Scrooge on the stage many times beginning in 1901, and again in a 1913 British silent film version.
</item 5>

<item 6>
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
</item 6>

<item 7>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 7>

<item 8>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 8>

<item 9>
Mr. Fezziwig
Mr. Fezziwig is a character from the novel "A Christmas Carol" created by Charles Dickens to provide contrast with Ebenezer Scrooge's attitudes towards business
ethics.

 Scrooge, who apprenticed under Fezziwig, is the very antithesis of the person he worked for as a young man.

 Mr. Fezziwig is portrayed as a jovial, foppish man with a large Welsh wig.

 In Act I, Scene 5 of "A Christmas Carol", the Ghost of Christmas Past takes Scrooge to revisit his youthful days in Fezziwig's world located at the cusp of the
Industrial Revolution.

 Dickens used Fezziwig to represent a set of communal values and a way of life which was quickly being swept away in the economic turmoil of the early
nineteenth century.
</item 9>

<item 10>
Scrooge, or, Marley's Ghost
Scrooge, or, Marley's Ghost is a 1901 British short silent drama film, directed by Walter R. Booth, featuring the miserly Ebenezer Scrooge confronted by
Marley's ghost and given visions of Christmas past, present, and future, is the earliest known film adaptation of Charles Dickens's 1843 novel "A Christmas
Carol".

 The film, "although somewhat flat and stage-bound to modern eyes," according to Ewan Davidson of BFI Screenonline, "was an ambitious undertaking at the time,"
as, "not only did it attempt to tell an 80 page story in five minutes, but it featured impressive trick effects, superimposing Marley's face over the door
knocker and the scenes from his youth over a black curtain in Scrooge's bedroom."
</item 10>
=== END ITEM id=8975a66e15f0c927 ===

=== ITEM id=2027cbaddb1f8ab9 ===
Gold sentences and facts:
Sentence 0: Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York, United States.
  Fact 0.0: Albany International Airport has the IATA code ALB.
  Fact 0.1: Albany International Airport has the ICAO code KALB.
  Fact 0.2: Albany International Airport has the FAA LID ALB.
  Fact 0.3: Albany International Airport is a public airport.
  Fact 0.4: Albany International Airport is seven miles (11 km) northwest of Albany.
  Fact 0.5: Albany International Airport is in Albany County, New York, United States.
Sentence 1: Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown Madison, the capital of Wisconsin.
  Fact 1.0: Dane County Regional Airport (DCRA) has the IATA code MSN.
  Fact 1.1: Dane County Regional Airport (DCRA) has the ICAO code KMSN.
  Fact 1.2: Dane County Regional Airport (DCRA) has the FAA LID MSN.
  Fact 1.3: Dane County Regional Airport is also known as Truax Field.
  Fact 1.4: Dane County Regional Airport is a civil-military airport.
  Fact 1.5: Dane County Regional Airport is located six miles northeast of downtown Madison.
  Fact 1.6: Madison is the capital of Wisconsin.

Retrieved items:
<item 1>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 1>

<item 2>
The entries describe U.S. and Saudi airports and authorities: Atlanta South (formerly Tara Field); remote Yakutat, served through Essential Air Service;
Wilmington, NC; Pennsylvania’s SARAA, overseeing four airports; Ta’if, a regional airport with international flights; Raleigh Exec, a reliever; Texarkana,
serving general aviation and American Eagle; Pensacola, despite its name, without direct international flights; Wilmington, Delaware; Columbus (formerly Port
Columbus); Amarillo, renamed for
</item 2>

<item 3>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 3>

<item 4>
Truax Field Air National Guard Base
Truax Field Air National Guard Base (IATA: MSN, ICAO: KMSN, FAA LID: MSN) , also known as Truax Field, is a military facility located at Dane County Regional
Airport.

 It is located five miles (8 km) northeast of the center of Madison, a city in Dane County, Wisconsin, United States.
</item 4>

<item 5>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 5>

<item 6>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 6>

<item 7>
Brainerd Lakes Regional Airport is a city- and county-owned, general-aviation airport near Brainerd, Minnesota, with one commercial airline. Grand Sky,
established in 2015 at Grand Forks Air Force Base, is a 217-acre UAS business park with runway access. Grand Forks Municipal Airport closed after World War II
and became urban land; Grand Forks International Airport serves the area and offers customs but no scheduled international flights. Nearby Emerado had 414
residents in
</item 7>

<item 8>
Air Wisconsin
Air Wisconsin Airlines is a regional airline based at Appleton International Airport in the town of Greenville, Wisconsin, United States, near Appleton.

 Air Wisconsin is the largest privately held regional airline in the United States.

 It currently operates regional jet flights as American Eagle under contract to American Airlines via a code sharing agreement, serving cities in the U.S. and
Canada with hubs at Boston Logan International Airport (BOS), Charlotte Douglas International Airport (CLT), Philadelphia International Airport (PHL) and Ronald
Reagan Washington National Airport (DCA) .

 Air Wisconsin previously operated United Express service on behalf of United Airlines followed by US Airways Express service on behalf of US Airways prior to
becoming an American Eagle regional air carrier.

 The company has announced it will once again partner with United Airlines as a United Express code sharing air carrier with primary hubs to be located at
Chicago O'Hare International Airport (ORD) and Washington Dulles International Airport (IAD) beginning in September 2017 and will be "exclusively operating" as
United Express by March 2018.
</item 8>

<item 9>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 9>

<item 10>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 10>
=== END ITEM id=2027cbaddb1f8ab9 ===

=== ITEM id=5494fead774add9f ===
Gold sentences and facts:
Sentence 0:  The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
  Fact 0.0: The documentary explores the history of Sunset Strip.
  Fact 0.1: Sunset Strip is a mile and a half long stretch of road.
  Fact 0.2: Sunset Strip runs through West Hollywood.
Sentence 1:  The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.
  Fact 1.0: The film features some early pioneers of the hardcore punk music scene.
  Fact 1.1: Bad Brains is an early hardcore punk pioneer featured in the film.
  Fact 1.2: Black Flag is an early hardcore punk pioneer featured in the film.
  Fact 1.3: D.O.A. is an early hardcore punk pioneer featured in the film.
  Fact 1.4: Minor Threat is an early hardcore punk pioneer featured in the film.
  Fact 1.5: Minutemen is an early hardcore punk pioneer featured in the film.
  Fact 1.6: SSD is an early hardcore punk pioneer featured in the film.
  Fact 1.7: The film features other early hardcore punk pioneers besides those named.

Retrieved items:
<item 1>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 1>

<item 2>
Riot on Sunset Strip
Riot on Sunset Strip is a 1967 counterculture-era exploitation movie, released by American International Pictures.

 It was filmed and released within four months of the late-1966 Sunset Strip curfew riots.
</item 2>

<item 3>
Sunset Strip
Sunset Strip is the mile-and-a-half (2.4 km) stretch of Sunset Boulevard that passes through West Hollywood, California.

 It extends from West Hollywood's eastern border with Hollywood at Crescent Heights Boulevard to its western border with Beverly Hills at Sierra Drive.

 Sunset Strip is probably the best-known portion of Sunset Boulevard, with boutiques, restaurants, rock clubs, and nightclubs.

 It is also known for its array of huge, colorful billboards.
</item 3>

<item 4>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 4>

<item 5>
Sunset Strip curfew riots
The Sunset Strip curfew riots, also known as the "hippie riots", were a series of early counterculture-era clashes that took place between police and young
people on the Sunset Strip in Hollywood, California, beginning in the mid-1966 and continuing on and off through the early 1970s.
</item 5>

<item 6>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 6>

<item 7>
Hellfest (American music festival)
Hellfest was an American hardcore punk festival.

 Named for the intense summer heat at the show, the festival originally began featuring hardcore bands before crossing into several genres later on.

 Hellfest was known for its fan-friendly atmosphere, (including a lack of barriers, providing easy access to the stage for stage diving) and for its high
social-awareness, focusing on such causes as animal rights, with many attendees following the straight edge and/or vegan lifestyle.

 It continued as an annual event until it was cancelled in 2005.

 After its cancellation less than thirty-six hours before doors were scheduled to open on the festival's ninth year, some fans, unhappy with the inadequate
processing of refunds, filed small claims court lawsuits against promoter Keith Allen.
</item 7>

<item 8>
Sunset Strip (2012 film)
Sunset Strip is a 2012 documentary directed by Hans Fjellestad, and produced by Tommy Alastra.

 The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
</item 8>

<item 9>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 9>

<item 10>
Mayor of the Sunset Strip
Mayor of the Sunset Strip is a 2003 documentary film on the life of Rodney Bingenheimer directed by George Hickenlooper, and produced by Chris Carter.

 In 2011, Craig Hlavaty of the "Houston Press" named "Mayor of the Sunset Strip" at number eight on the paper's list of "The 31 Best Music Documentaries of All
Time".

 The film won the Best Documentary Feature at the 2004 Santa Barbara International Film Festival.
</item 10>
=== END ITEM id=5494fead774add9f ===

=== ITEM id=721fa84abd3d83f8 ===
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
Emily's Quest
Emily's Quest is a novel and the last of the "Emily" trilogy by Lucy Maud Montgomery.

 After finishing "Emily Climbs", Montgomery suspended writing "Emily's Quest" and published "The Blue Castle"; she resumed writing and published in 1927.
</item 2>

<item 3>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 3>

<item 4>
Mabel May
Henrietta Mabel May (September 11, 1877 – October 8, 1971), was a Canadian artist in the early 20th century and an organizer of women artists.

 Based in Quebec early in her career, and later in her life, Vancouver, she was a well-known painter and member of multiple important Canadian artist groups,
including the Art Association of Montreal, the Beaver Hall Group and the Canadian Group of Painters.

 Her works have been displayed at the Canadian War Memorial, National Gallery, the Vancouver Art Gallery and many smaller galleries throughout Quebec.

 She has been commonly referred to as the “Emily Carr of Montreal” due to her interest in landscape and nature focused artwork.

 Her artwork was mostly influenced by her avid interest in French Impressionists.
</item 4>

<item 5>
Anne of Green Gables
Anne of Green Gables is a 1908 novel by Canadian author Lucy Maud Montgomery (published as L. M. Montgomery).

 Written for all ages, it has been considered a children's novel since the mid-twentieth century.

 It recounts the adventures of Anne Shirley, an 11-year-old orphan girl who is mistakenly sent to Matthew and Marilla Cuthbert, a middle-aged brother and sister
who had intended to adopt a boy to help them on their farm in the fictional town of Avonlea on Prince Edward Island.

 The novel recounts how Anne makes her way with the Cuthberts, in school, and within the town.
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
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 9>

<item 10>
Rosario Castellanos
Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.

 Along with the other members of the Generation of 1950 (the poets who wrote following the Second World War, influenced by César Vallejo and others), she was
one of Mexico's most important literary voices in the last century.

 Throughout her life, she wrote eloquently about issues of cultural and gender oppression, and her work has influenced feminist theory and cultural studies.

 Though she died young, she opened the door of Mexican literature to women, and left a legacy that still resonates today.
</item 10>
=== END ITEM id=721fa84abd3d83f8 ===

=== ITEM id=fbe808ab539feb38 ===
Gold sentences and facts:
Sentence 0:  The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover "Come Together".
  Fact 0.0: The tour accompanied the release of the piano-based album 'In the Running'.
  Fact 0.1: The tour featured pared-back versions of his hits.
  Fact 0.2: The tour featured pared-back versions of his album tracks.
  Fact 0.3: The tour featured a pared-back version of The Beatles cover 'Come Together'.
Sentence 1:  The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
  Fact 1.0: The song reached the top of the charts in the United States.
  Fact 1.1: The song peaked at No. 4 in the United Kingdom.

Retrieved items:
<item 1>
Range Life (song)
The single was not commercially released in the USA; it was issued by the band's UK label at the time, Big Cat.

 Both B-sides are outtakes from the "Crooked Rain, Crooked Rain" sessions and are included on the 2004 of that album.

 This song was one of many to be included in the group's greatest hits album "".
</item 1>

<item 2>
Baby, Stop Crying
"Baby, Stop Crying" is a song written by Bob Dylan, released in the summer of 1978 as a single and in a longer album version on "Street Legal".

 It was performed live only during his big-band worldwide tour of 1978.

 The song charted at #13 in the UK and was a top-ten song in much of Europe, although it failed to chart in the United States.
</item 2>

<item 3>
On Top of Old Smoky
"On Top of Old Smokey" is a traditional folk song of the United States.

 As recorded by The Weavers, the song reached the pop music charts in 1951.

 It is catalogued as Roud Folk Song Index No. 414.

 In one version the first verse is the following; for more on the words see below.
</item 3>

<item 4>
One Acoustic Night
One Acoustic Night is the third album live by the German hard rock band Bonfire.

 It was released in 2005 by E-M-S Music and is a two-disc album that is a live acoustic performance in Germany.

 It also features a studio song dedicated to the victims of the tsunami disaster in Thailand, recorded in English, German and Spanish.

 Chris Limburg and Thomas Streck were guest musicians that helped with the live performance, which is all in German.

 A DVD was released at the same time of the acoustic performance.
</item 4>

<item 5>
Basie on the Beatles
Basie on the Beatles is an album by pianist and bandleader Count Basie featuring performances recorded in late 1969 and released on the short-lived Happy Tiger
label.

 It was Basie's second album of Beatles' compositions following 1966's "Basie's Beatle Bag" and featured liner notes by Ringo Starr.
</item 5>

<item 6>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 6>

<item 7>
Live Acoustic America
Live Acoustic America is an acoustic live recording of classic Howard Jones songs, released in 1996.

 It was recorded in Los Angeles to a sell out crowd and features Carol Steele on percussion.

 The acoustic tour travelled the world, with Jones playing venues in Europe, the USA and Jamaica.

 The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover
"Come Together".
</item 7>

<item 8>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 8>

<item 9>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 9>

<item 10>
Beatles (song)
"Beatles", written by Sven Olov Bagge and Claes Bure, was the song that, performed by Swedish dansband Forbes, won the Swedish Melodifestivalen 1977.

 The song's lyrics are about the popular 1960s British rock group the Beatles and include the phrase "Yeah, yeah, yeah" as an homage to the Beatles song "She
Loves You".

 Forbes took the song to the United Kingdom for the Eurovision Song Contest 1977, with Anders Berglund as conductor.

 The audience cheered at the song, not unexpectedly since the contest took place in the United Kingdom, but the jury had another opinion.

 The song finished last, with only two points, from West Germany.
</item 10>
=== END ITEM id=fbe808ab539feb38 ===

=== ITEM id=ff1d2497431af728 ===
Gold sentences and facts:
Sentence 0: Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
  Fact 0.0: Ernst Willi Messerschmid is Prof. Dr.
  Fact 0.1: Ernst Willi Messerschmid was born on May 21, 1945.
  Fact 0.2: Ernst Willi Messerschmid is a German physicist.
  Fact 0.3: Ernst Willi Messerschmid is a former astronaut.
Sentence 1: Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
  Fact 1.0: Philippe Perrin was born on January 6, 1963.
  Fact 1.1: Philippe Perrin is a Colonel in the French Air Force.
  Fact 1.2: Philippe Perrin is a French test pilot.
  Fact 1.3: Philippe Perrin is a former CNES astronaut.
  Fact 1.4: Philippe Perrin is a former European Space Agency astronaut.

Retrieved items:
<item 1>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 1>

<item 2>
Philippe Perrin (artist)
Philippe Perrin, (La Tronche, 10 August 1964) is a French artist and contemporary photographer who lives and works in Paris.
</item 2>

<item 3>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 3>

<item 4>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 4>

<item 5>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 5>

<item 6>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 6>

<item 7>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 7>

<item 8>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 8>

<item 9>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 9>

<item 10>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 10>
=== END ITEM id=ff1d2497431af728 ===

=== ITEM id=a64df306510bf99d ===
Gold sentences and facts:
Sentence 0: Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.
  Fact 0.0: Frederick King Keller was born in 1954.
  Fact 0.1: Frederick King Keller was born in Buffalo, New York.
  Fact 0.2: Frederick King Keller is an American director.
  Fact 0.3: Frederick King Keller is an American producer.
  Fact 0.4: Frederick King Keller is an American screenwriter.
  Fact 0.5: Frederick King Keller works in film and television.
Sentence 1: Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
  Fact 1.0: Ian Schneider was born on October 3, 1992.
  Fact 1.1: Ian Schneider is an independent American film director.
  Fact 1.2: Ian Schneider is an American screenwriter.

Retrieved items:
<item 1>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 1>

<item 2>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 2>

<item 3>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 3>

<item 4>
Colby Keller
Colby Keller (born October 18, 1980) is an American visual artist, blogger, and pornographic film actor.

 His career in porn film started in 2004 at Sean Cody and has since expanded to include such studios as Cocksure Men, Randy Blue, Titan Men, Falcon, CockyBoys
and Men.com.

 Keller has amassed a large fan base with both his films and his long-standing blog, "The Big Shoe Diaries".

 He has been nominated for a number of awards and in 2013, World of Wonder awarded him the WOWie Award for Best Blog.

 Keller is nominated for 8 2015 Grabby Awards.

 Keller has also appeared in the short, Zolushka, a queer retelling of Cinderella and the popular series Capitol Hill both by the cult filmmaker Wes Hurley.
</item 4>

<item 5>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 5>

<item 6>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 6>

<item 7>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 7>

<item 8>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 8>

<item 9>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 9>

<item 10>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 10>
=== END ITEM id=a64df306510bf99d ===

=== ITEM id=0d06adee2f8d14ff ===
Gold sentences and facts:
Sentence 0: Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.
  Fact 0.0: Midland F1 Racing was a Formula One constructor.
  Fact 0.1: Midland F1 Racing was a racing team.
  Fact 0.2: Midland F1 Racing is also known as MF1 Racing.
Sentence 1:  In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
  Fact 1.0: In 2007, the team competed as Spyker F1.
  Fact 1.1: In 2008, the team was sold to Indian businessman Vijay Mallya.
  Fact 1.2: In 2008, the team was renamed Force India F1.
Sentence 2: The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.
  Fact 2.0: The Spyker F1 Team was known as the Etihad Aldar Spyker F1 Team for sponsorship reasons.
  Fact 2.1: The Spyker F1 Team was a Formula One team.
  Fact 2.2: The Spyker F1 Team competed in the 2007 Formula One World Championship.
  Fact 2.3: The Spyker F1 Team was created by Spyker Cars after their buyout of the short-lived Midland F1 team.
  Fact 2.4: Midland F1 was formerly known as Jordan Grand Prix.

Retrieved items:
<item 1>
Jordan 195
The Jordan 195 was a Formula One car for the 1995 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 The team never employed a test driver.

 The engine was a Peugeot A10 3.0 V10.

 The team's main sponsor was Total.
</item 1>

<item 2>
Spyker F8-VII
The Spyker F8-VII (subsequently known as the Force India VJM01) was a Formula One car, constructed by Spyker F1 that competed in the 2007 Formula One World
Championship.

 A "B Specification" car named the Spyker F8-VIIB was launched at the Italian Grand Prix and used for the remainder of the 2007 season.

 The engine of Spyker F8-VII car was Ferrari 056 despite the team opted for 2006-spec engine due to cost reasons.

 For the 2008 World Championship, Force India used a slightly modified version of the F8-VIIB, called the Force India VJM01 named after team owners Vijay
Mallya, Jan Mol and Michiel Mol.

 The VJM01 used 2007-spec Ferrari 056 engines instead of 2008-spec.

 The F8-VII was the only car constructed by Spyker F1 in their own right after their take-over from the struggling Midland F1 team part way through 2006.
</item 2>

<item 3>
Midland F1 Racing
Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.

 It competed in the 2006 Formula One season with drivers Christijan Albers and Tiago Monteiro.

 The team was created by the renaming of Jordan Grand Prix after its purchase by Canadian businessman, and owner of the Midland Group, Alex Shnaider.

 The team was registered as the first Russian Formula One team, reflecting Shnaider's roots, although it continued to be based in the United Kingdom, at
Jordan's Silverstone factory.

 Towards the end of the 2006 season, the team was sold to Spyker Cars N.V.; the team raced in its last three Grands Prix under the official name "Spyker MF1
Racing".

 In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
</item 3>

<item 4>
Simtek
Simtek (Simulation Technology) was an engineering consultancy firm and Formula One racing team.

 The Formula One (F1) engineering consultancy arm, Simtek Research, was founded in 1989 by Max Mosley and Nick Wirth.

 It originally was involved in many areas of Formula One, including wind tunnel construction and chassis building for third parties.

 Simtek Grand Prix, the racing team, launched in 1993 and competed in the 1994 and 1995 seasons achieving a best result of ninth place.

 With large debts and a lack of sponsorship money, Simtek went into voluntary liquidation in June 1995.
</item 4>

<item 5>
Marussia F1
The Marussia F1 Team (subsequently Manor Marussia F1 Team) was an Anglo-Russian Formula One racing team and constructor which was based in Banbury, Oxfordshire
in the United Kingdom.

 The team was operated by Manor Motorsport (formerly Marussia Manor Racing), which was previously a subsidiary of Marussia Motors, a now defunct sports car
manufacturer which was based in Moscow.

 The team originally started racing in under the "Virgin Racing" name; the following year Virgin adopted Marussia as a title sponsor becoming "Marussia Virgin
Racing" until being fully rebranded as the "Marussia F1 Team" for .
</item 5>

<item 6>
The Triumph Street Triple is a Triumph naked motorcycle introduced in late 2007. Midland F1, formed when Alex Shnaider bought Jordan, raced in 2006 with
Christijan Albers and Tiago Monteiro; sold to Spyker late that year, it became Spyker F1 in 2007, then Force India in 2008. Spyker’s orange-and-silver F8-VII
used Ferrari engines; its updated VJM01 served Force India in 200
</item 6>

<item 7>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 7>

<item 8>
Speedway Motorsports
Speedway Motorsports, Inc. (SMI) is an American corporation that owns and manages racing facilities that host NASCAR, IndyCar Series, NHRA, World of Outlaws and
other motor racing series.

 The company's headquarters are located at Charlotte Motor Speedway, in the suburbs of Charlotte, NC.

 SMI owns nine racing facilities with a combined seating capacity of approximately 885,000.

 In addition to operating racetracks, SMI owns Performance Racing Network (PRN), U.S. Legends Cars International, and co-owns Motorsports Authentics.
</item 8>

<item 9>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 9>

<item 10>
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 10>
=== END ITEM id=0d06adee2f8d14ff ===

=== ITEM id=51d1c695e7621702 ===
Gold sentences and facts:
Sentence 0: Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.
  Fact 0.0: Pennsylvania Route 145 (PA 145) is a state highway.
  Fact 0.1: Pennsylvania Route 145 is 20.89 miles long.
  Fact 0.2: Pennsylvania Route 145 runs north-south.
  Fact 0.3: Pennsylvania Route 145 is in the Lehigh Valley area of eastern Pennsylvania.
Sentence 1:  MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the remainder of the road a four-lane divided highway.
  Fact 1.0: MacArthur Road is a divided highway.
  Fact 1.1: Between U.S. Route 22 (US 22) and Eberhart Road, MacArthur Road is six lanes wide.
  Fact 1.2: Between U.S. Route 22 and Eberhart Road, MacArthur Road has a Jersey barrier.
  Fact 1.3: Between U.S. Route 22 and Eberhart Road, MacArthur Road has jughandles.
  Fact 1.4: The remainder of MacArthur Road is a four-lane divided highway.
Sentence 2:  Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.
  Fact 2.0: Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.

Retrieved items:
<item 1>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 1>

<item 2>
Breadon Field
Breadon Field was a minor league ballpark in Whitehall Township, Pennsylvania, located on the east side of MacArthur Road, about 0.5 mile north of the U.S.
Route 22 interchange.
</item 2>

<item 3>
Pennsylvania Canal (Delaware Division)
as part of the solution to the United States' first energy crisis.

 The Commonwealth of Pennsylvania built the Delaware canal to feed anthracite stone coal to energy-hungry Philadelphia as part of its transportation
infrastructure building plan known as the Main Line of Public Works—a legislative initiative creating a collection of self-reinforcing internal improvements to
commercial transportation capabilities.
</item 3>

<item 4>
Pennsylvania Route 145
MacArthur Road is the location of the main commercial center of the Lehigh Valley.

 North of Eagle Point, PA 145 becomes a two-lane undivided road that parallels the Lehigh River, crossing the river into Northampton County at Treichlers.

 The route continues along the east bank of the river and passes through Walnutport before reaching its northern terminus.

 PA 145 is dedicated as the Battle of the Bulge Veterans Memorial Highway in honor of the veterans who fought in the Battle of the Bulge.
</item 4>

<item 5>
Susquehanna and Tidewater Canal
A copy of a detailed survey blueprint of the entire canal system including structures and property ownership details was donated (date unknown) by the Safe
Harbor Water Power Corporation to Millersville University (assessed on 10/06/2016 in the MU Archives at Sp.

 Coll.

 Map 386.409748 S128 Folio).

 The survey consists of 67 pages, 98 x 30 cm.

 and is undated but the assumption is the original was created while the canal was in use.
</item 5>

<item 6>
The entries cover historic U.S. canals, waterways, water systems, and sites. Transport canals include Beaver–Erie (136 miles), Delaware (60), Whitewater (76),
and Susquehanna–Tidewater (43), plus interstate and regional routes; many declined with railroad competition. The Dismal Swamp Canal, opened in 1805 and closed
in 2016, served the Intracoastal Waterway; nearby park and refuge protect the swamp.
</item 6>

<item 7>
Pennsylvania Route 145
Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.

 It connects Interstate 78 (I-78) and PA 309 in Lanark, Lehigh County to PA 248 in Weiders Crossing, Northampton County.

 PA 145 is the main north-south arterial into Allentown, the third-largest city in the state.

 The route enters the city on South 4th Street and follows multiple streets to downtown, where it follows the one-way pair of 6th Street northbound and 7th
Street southbound.

 North of Allentown in Whitehall Township, a seven-mile (11.2 km) portion of PA 145 is known as MacArthur Road, named in honor of General Douglas MacArthur.

 MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the
remainder of the road a four-lane divided highway.

 MacArthur Road is the location of the main commercial center of the Lehigh Valley.
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
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 9>

<item 10>
Jersey barrier
A Jersey barrier or Jersey wall is a modular concrete or plastic barrier employed to separate lanes of traffic.

 It is designed to minimize vehicle damage in cases of incidental contact while still preventing the crossover case of a head-on collision.

 Jersey barriers are also used to reroute traffic and protect pedestrians and workers during highway construction, as well as temporary and semi-permanent
protections against landborne attack such as suicide vehicle bombs.

 A Jersey barrier is also known in the western United States as K-rail, or K-wall, a term borrowed from the California Department of Transportation
specification for temporary concrete traffic barriers, or colloquially as a Jersey bump.

 Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.
</item 10>
=== END ITEM id=51d1c695e7621702 ===
