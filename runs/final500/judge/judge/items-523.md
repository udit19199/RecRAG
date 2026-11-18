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

=== ITEM id=cb4e8f04b2c7c5da ===
Gold sentences and facts:
Sentence 0: Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an SS-"Obergruppenführer".
  Fact 0.0: Karl Hermann Frank's life dates are 24 January 1898 to 22 May 1946.
  Fact 0.1: Karl Hermann Frank was a prominent Sudeten German Nazi official in Czechoslovakia.
  Fact 0.2: Karl Hermann Frank was a Nazi official in Czechoslovakia prior to World War II.
  Fact 0.3: Karl Hermann Frank was a Nazi official in Czechoslovakia during World War II.
  Fact 0.4: Karl Hermann Frank was an SS-Obergruppenführer.
Sentence 1:  Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".
  Fact 1.0: Until April 1942, the Obergruppenführer rank was the highest commissioned SS rank.
  Fact 1.1: Until April 1942, the Obergruppenführer rank was inferior only to Reichsführer-SS.
  Fact 1.2: Reichsführer-SS was the SS rank held by Heinrich Himmler.
  Fact 1.3: RFSS was the internal SS abbreviation for Himmler.
  Fact 1.4: Obergruppenführer translates as senior group leader.
  Fact 1.5: The rank of Obergruppenführer was senior to Gruppenführer.

Retrieved items:
<item 1>
Field marshal
Field marshal (or field-marshal, abbreviated as FM) ( , ) is a very senior military rank, ordinarily senior to the general officer ranks.

 Usually it is the highest rank in an army, and when it is, few (if any) persons are appointed to it.

 It is considered as a five-star rank (OF-10) in modern-day armed forces in many countries.

 Promotion to the rank of field marshal in many countries historically required extraordinary military achievement by a general (a wartime victory).

 However, the rank has also been used as a divisional command rank and also as a brigade command rank.

 Examples of the different uses of the rank include Austria-Hungary, Prussia and Germany for an extraordinary achievement; Spain and Mexico for a divisional
command (Spanish: "mariscal de campo" ); and France, Portugal and Brazil for a brigade command (French: "maréchal de camp" , Portuguese: "marechal de campo" ).
</item 1>

<item 2>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 2>

<item 3>
Karl Hermann Frank
Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an
SS-"Obergruppenführer".

 He was tried, convicted and executed after World War II for his role in organizing the massacres of the people of the Czech villages of Lidice and Ležáky.
</item 3>

<item 4>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 4>

<item 5>
Frank A. Armstrong
Frank Alton Armstrong Jr. (May 24, 1902 – August 20, 1969) was a lieutenant general of the United States Air Force.

 As a brigadier general in the United States Army Air Forces during World War II, he was the inspiration for the main character in the novel and subsequent
film, "Twelve O'Clock High."

 After the war, he held a variety of senior leadership positions prior to and following the establishment of the USAF as an independent service in 1947.

 Promoted to major general in 1950, he advanced to lieutenant general in 1956 and retired at that rank in 1962.
</item 5>

<item 6>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 6>

<item 7>
Obergruppenführer
Obergruppenführer (] , "senior group leader") was a Nazi Party paramilitary rank that was first created in 1932 as a rank of the "Sturmabteilung" (SA), and
adopted by the "Schutzstaffel" (SS) one year later.

 Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation
for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".

 A similarly named rank of "Untergruppenführer" existed in the SA from 1929 to 1930 and as a title until 1933.

 In April 1942, the new rank of "SS-Oberst-Gruppenführer" was created which was above "Obergruppenführer" and below "Reichsführer-SS".
</item 7>

<item 8>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 8>

<item 9>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 9>

<item 10>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 10>
=== END ITEM id=cb4e8f04b2c7c5da ===

=== ITEM id=861e6cf5df84f4cf ===
Gold sentences and facts:
Sentence 0: Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to 2003.
  Fact 0.0: Václav Havel was born on 5 October 1936.
  Fact 0.1: Václav Havel died on 18 December 2011.
  Fact 0.2: Václav Havel was a Czech writer.
  Fact 0.3: Václav Havel was a former dissident.
  Fact 0.4: Václav Havel served as the 1st President of the Czech Republic.
  Fact 0.5: Václav Havel served as President of the Czech Republic from 1993 to 2003.
Sentence 1: Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.
  Fact 1.0: Penelope Fitzgerald was born on 17 December 1916.
  Fact 1.1: Penelope Fitzgerald died on 28 April 2000.
  Fact 1.2: Penelope Fitzgerald was English.
  Fact 1.3: Penelope Fitzgerald was a Booker Prize-winning novelist.
  Fact 1.4: Penelope Fitzgerald was a poet.
  Fact 1.5: Penelope Fitzgerald was an essayist.
  Fact 1.6: Penelope Fitzgerald was a biographer.

Retrieved items:
<item 1>
Vaněk plays
The Vaněk plays are a set of plays to which the character Ferdinand Vaněk is central.

 Vaněk first appeared in the play "Audience" by Václav Havel.

 He subsequently appeared in three other plays by Havel ("Protest", "Unveiling", and "Dozens of Cousins"), as well as plays by his friends and colleagues,
including Pavel Landovský and Tom Stoppard.
</item 1>

<item 2>
Václav Havel Tribute Concert
Tribute to Václav Havel (Czech: Pocta Václavu Havlovi) was an event held in memory of Václav Havel, the last Czechoslovak and the first Czech President, writer,
playwright and human rights activist.

 The concert took place in Lucerna Music Bar in Prague on 23 December 2011.

 Václav Havel died on Sunday 18 December 2011 at 10:15 a.m.
</item 2>

<item 3>
Czech presidential election, 1993
The 1993 Czech presidential election was held on 26 January 1993 to elect the first President of independent Czech Republic.

 The first election was done indirectly by the members of the Czech Republic Parliament.

 Václav Havel was elected President.

 The election was complicated only by attacks of Republicans against Havel and by bomb threat to the Parliament.
</item 3>

<item 4>
Václav Havel Human Rights Prize
The Václav Havel Human Rights Prize is an annual €60,000 award established in 2013 by the Parliamentary Assembly of the Council of Europe, the Václav Havel
Library and the Charta 77 Foundation, to honour "outstanding" civil society action in defence of human rights, in Europe and beyond.
</item 4>

<item 5>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
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
Letters to Olga
Letters to Olga ("Czech:Dopisy Olze") is a book of compiled letters written by Czech playwright, dissident, and future president, Václav Havel to his wife Olga
Havlová during his nearly four-year imprisonment from May 1979 to March 1983.

 (Havel was released when he came down with a high fever and received a medical discharge).

 Havel was imprisoned by the communist regime of then Czechoslovakia for being one of the leaders of The Committee for the Defense of the Unjustly Prosecuted
(VONS) - most of whom had been signatories of the human rights document Charter 77.
</item 7>

<item 8>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 8>

<item 9>
Theatre Organization of Cyprus (THOC), founded in 1970, promotes theatre and international artistic relations; it is governed by a presidentially appointed
board. Copenhagen’s Tivoli Concert Hall, designed by Frits Schlegel and Hans Hansen, opened in 1956. Czech dissident, playwright and writer Václav Havel became
Czechoslovakia’s last president (1989–92) and the Czech Republic’s first (1993–2003). A
</item 9>

<item 10>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 10>
=== END ITEM id=861e6cf5df84f4cf ===

=== ITEM id=474046c6b2f33ad6 ===
Gold sentences and facts:
Sentence 0: The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a unique numeric identifier, referred to as a "DUNS number" to a single business entity.
  Fact 0.0: The Data Universal Numbering System is abbreviated as DUNS or D-U-N-S.
  Fact 0.1: The Data Universal Numbering System is a proprietary system.
  Fact 0.2: The Data Universal Numbering System is developed by Dun & Bradstreet (D&B).
  Fact 0.3: The Data Universal Numbering System is regulated by Dun & Bradstreet (D&B).
  Fact 0.4: The Data Universal Numbering System assigns a unique numeric identifier to a single business entity.
  Fact 0.5: The unique numeric identifier is referred to as a DUNS number.
Sentence 1:  Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
  Fact 1.0: Dun & Bradstreet is often referred to as D&B.
  Fact 1.1: The company's database contains more than 265 million business records.
  Fact 1.2: The company's database contains business records worldwide.

Retrieved items:
<item 1>
Data Universal Numbering System
The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a
unique numeric identifier, referred to as a "DUNS number" to a single business entity.

 It was introduced in 1963 to support D&B's credit reporting practice.

 It is standard worldwide.

 DUNS users include the European Commission, the United Nations, and the United States government.

 More than 50 global industry and trade associations recognize, recommend, or require DUNS.

 The DUNS database contains over 250 million entries for businesses throughout the world.
</item 1>

<item 2>
Graph query row:
entity: Data Universal Numbering System
relationship: CREATED_BY
related: Dun & Bradstreet
</item 2>

<item 3>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 3>

<item 4>
Calypso Ichthyological Database
The Calypso Ichthyological Database numbering system is an open source free repository allowing the unique identification numbering of all fish species with a
six-numeral fixed number.

 This number remains the same throughout any alterations or taxonomic changes to the species' accepted current scientific name and allows for recording of
species data in 32 unique data fields including all previous names in 32 languages.

 It has been in use since 1994 and was pioneered by its inventor, Gerald H. Jennings, with the technical assistance of Terry Hall.

 It is accompanied by a vast photographic library of fish species and is free at the point of use to all researchers and academics.

 Much of the data is also available in published format and online.
</item 4>

<item 5>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 5>

<item 6>
Universal Numbering System
The Universal Numbering System is a dental notation system for associating information to a specific tooth, and is commonly used in the United Kingdom.
</item 6>

<item 7>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 7>

<item 8>
United Nations (band)
The idea of the side project can be traced back to the early 2000s, but it was not until 2008 that United Nations took shape.

 Geoff Rickly has said the band started over tequila with Daryl Palumbo.

 At the start of the band, all members except for Rickly were under contracts with other record labels and were not legally permitted to work in projects
released by another, making it difficult for the official lineup of the band to be well known.

 However some members from the band's original line-up have been revealed through interviews and social media posts, such as Daryl Palumbo, Jonah Bayer and
Lukas Previn.

 Since Rickly was the only member whose primary band was not under contract, his name is the only one that could legally appear in press materials.

 Press photos of the band released in 2008 depict four people wearing Ronald Reagan masks.
</item 8>

<item 9>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 9>

<item 10>
Broward County Uniform Station Numbering
The Broward County Uniform Station Numbering system is a system by which all fire stations and fire apparatus in Broward County are numbered.

 The numbering system was developed by the Fire Chief's Association of Broward County in order to minimize confusion and create efficient radio communications
at large fires or emergency scenes, where multiple departments were working together.

 The system went into effect on October 1, 1990 after a large fire at GLS Fiberglass in Port Everglades in 1988.

 Prior to 1990, each fire department had its own numbering system, usually starting at 1 and proceeding to number each station that department operated from.

 At this large fire in Port Everglades, Dania Beach Fire Department, Hollywood Fire Department, Port Everglades Public Safety, and Fort Lauderdale Fire
Department all were on scene with their own Engine 1s.

 As a result, every time a person called for "Engine 1" on fireground radios, confusion would ensue.
</item 10>
=== END ITEM id=474046c6b2f33ad6 ===

=== ITEM id=14a68e26c46eb4c8 ===
Gold sentences and facts:
Sentence 0: Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.
  Fact 0.0: Jürgen Melzer was born on 22 May 1981.
  Fact 0.1: Jürgen Melzer was born in Vienna.
  Fact 0.2: Jürgen Melzer is an Austrian tennis player.
Sentence 1: Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
  Fact 1.0: Tom Gullikson was born on September 8, 1951.
  Fact 1.1: Tom Gullikson is a tennis coach.
  Fact 1.2: Tom Gullikson is a former professional tennis player.
  Fact 1.3: Tom Gullikson is from the United States.

Retrieved items:
<item 1>
Julian Knowle
Julian Knowle (born 29 April 1974) is an Austrian male professional tennis player.

 Being a born left-hander, Knowle is now one of the few on the ATP Tour who plays his forehand, backhand, and even volleys double-handed.

 He was Austria's most successful doubles player in history by reaching world no. 6 in the ATP doubles rankings in January 2008, before being matched by Jürgen
Melzer, who reached no. 6 in September 2010, and overtaken by Alexander Peya, who reached no. 3 in August 2013.
</item 1>

<item 2>
Tom Gullikson
Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
</item 2>

<item 3>
2010 Pilot Pen Tennis – Men's Doubles
Julian Knowle and Jürgen Melzer were the defending champions, but Melzer chose not to participate this year.As a result, Knowle partnered with Andy Ram, but
they lost to Robert Lindstedt and Horia Tecău in the semifinals.

 Lindstedt and Tecău went on to win the tournament, after defeating Rohan Bopanna and Aisam-ul-Haq Qureshi in the final 6–4, 7–5.
</item 3>

<item 4>
Ernests Gulbis
Ernests Gulbis (] , born 30 August 1988, and nicknamed "The Gull" or "Ernie") is a Latvian professional tennis player.

 In 2008, Gulbis won his first ATP Tour doubles title at the U.S. Men's Clay Court Championships, teaming with Rainer Schüttler, and in 2010 won his first ATP
Tour singles title in the Delray Beach, defeating Ivo Karlović in the final.

 In total, Gulbis has 6 ATP titles to his name and has never lost a final.

 His best performance at a Grand Slam is reaching the semifinals of the 2014 French Open.

 He had previously reached the quarterfinals of the 2008 French Open.

 Gulbis' career-high singles ranking is world No. 10, making him the only Latvian tennis player ever to be ranked inside the top 10 in ATP Singles Ranking.

 He achieved this in June 2014.
</item 4>

<item 5>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 5>

<item 6>
Graph query row:
person: Jürgen Melzer
nationality: Austrian
professional_status: tennis player
</item 6>

<item 7>
2012 US Open – Men's Doubles
Jürgen Melzer and Philipp Petzschner were the defending champions, but lost in the second round to Jamie Delgado and Ken Skupski.

 Melzer and Petzschner needed a wildcard to compete because they forgot to sign up.

<br>
</item 7>

<item 8>
Graph query row:
person: Tom Gullikson
nationality: None
professional_status: ['tennis coach', 'former professional tennis player']
</item 8>

<item 9>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 9>

<item 10>
1983 Bristol Open – Doubles
Tim Gullikson and Tom Gullikson were the defending champions, but Tim Gullikson did not participate this year.

 Tom Gullikson partnered Johan Kriek, losing in the final.
</item 10>
=== END ITEM id=14a68e26c46eb4c8 ===

=== ITEM id=6bde2d39db55b4fd ===
Gold sentences and facts:
Sentence 0: Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.
  Fact 0.0: The Dogo Cubano is also known as the Cuban Mastiff.
  Fact 0.1: The Dogo Cubano is also known as the Cuban Dogo.
  Fact 0.2: The Dogo Cubano is also known as the Cuban Dogge.
  Fact 0.3: The Dogo Cubano is an extinct dog breed.
  Fact 0.4: The Dogo Cubano is a dog breed from Cuba.
Sentence 1: The Dutch Shepherd is a herding dog of Dutch origin.
  Fact 1.0: The Dutch Shepherd is a herding dog.
  Fact 1.1: The Dutch Shepherd is a herding dog of Dutch origin.

Retrieved items:
<item 1>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 1>

<item 2>
Cuban Criollo
The Criollo Cubano originates from Spanish horses brought to Cuba by Diego Velázquez in 1751.

 There are four different breeds known as criollo: the Cubano de Paso, the Pinto Cubano, the Criollo de Trote and the Patibarcino.

 It is a small stout breed used by the Guajiro people.
</item 2>

<item 3>
Podenco Canario
Podenco Canario (In English: "Canary Islands Hound", "Canarian Warren Hound") is a breed of dog originally from the Canary Islands.

 The Podenco Canario is still used in the Canary Islands today in packs as a hunting dog, primarily used for rabbit hunting.

 The word "podenco" is Spanish for "hound".

 "Canario" means "Canarian" or "of the Canaries".
</item 3>

<item 4>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 4>

<item 5>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 5>

<item 6>
Romanian Raven Shepherd Dog
The Romanian Raven Shepherd Dog is a very large Romanian livestock guardian dog, taxonomized within the second group of dog breeds - Pinscher and Schanuzer -
Molossoid Breeds - Swiss Mountain and Cattle Dogs, section 2.2: Molossoid Breeds - Mountain type.
</item 6>

<item 7>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 7>

<item 8>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 8>

<item 9>
Dogo Cubano
Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.

 It was of the Bull Mastiff type and was used for dog fighting.
</item 9>

<item 10>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 10>
=== END ITEM id=6bde2d39db55b4fd ===

=== ITEM id=dfa810df58f9be9f ===
Gold sentences and facts:
Sentence 0: Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
  Fact 0.0: Milium's vernacular name is milletgrass.
  Fact 0.1: Milium is a genus of plants in the grass family.
  Fact 0.2: Milium plants are Eurasian, North American, and North African.
Sentence 1: Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.
  Fact 1.0: Eucommia is a genus of small trees.
  Fact 1.1: Eucommia trees are now native to China.
  Fact 1.2: The fossil record of Eucommia shows a much wider distribution.
Sentence 2:  The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in herbology such as traditional Chinese medicine.
  Fact 2.0: Eucommia has a single living species, Eucommia ulmoides.
  Fact 2.1: Eucommia ulmoides is near threatened in the wild.
  Fact 2.2: Eucommia ulmoides is widely cultivated in China for its bark.
  Fact 2.3: Eucommia ulmoides is highly valued in herbology, such as traditional Chinese medicine.

Retrieved items:
<item 1>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 1>

<item 2>
Ocimum
Ocimum is a genus of aromatic annual and perennial herbs and shrubs in the family Lamiaceae, native to the tropical and warm temperate regions of all 6
inhabited continents, with the greatest number of species in Africa.

 Its best known species are the cooking herb Cooking basil, "O. basilicum" and the medicinal herb Tulsi (holy basil), "O. tenuiflorum".
</item 2>

<item 3>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 3>

<item 4>
Milium (plant)
Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
</item 4>

<item 5>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 5>

<item 6>
Genoplesium
Genoplesium commonly known as midge orchids, is a genus of about 50 species of flowering plants in the orchid family, Orchidaceae and is found in Australia, New
Zealand and New Caledonia.

 Midge orchids are terrestrial herbs with a single leaf at the base of the plant.

 They are similar to orchids in the genus "Prasophyllum" in that plants without flowers have a hollow, onion-like leaf.

 The flowers are small but often scented and attractive to their insect pollinators.

 There is disagreement about which species belong to this genus and some taxonomists suggest that most belong in the genus "Corunastylis".
</item 6>

<item 7>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 7>

<item 8>
Wild boar
The wild boar ("Sus scrofa"), also known as the wild swine or Eurasian wild pig, is a suid native to much of Eurasia, North Africa, and the Greater Sunda
Islands.

 Human intervention has spread its distribution further, making the species one of the widest-ranging mammals in the world, as well as the most widely spread
suiform.

 Its wide range, high numbers, and adaptability mean that it is classed as least concern by the IUCN and it has become an invasive species in part of its
introduced range.

 The animal probably originated in Southeast Asia during the Early Pleistocene, and outcompeted other suid species as it spread throughout the Old World.
</item 8>

<item 9>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 9>

<item 10>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 10>
=== END ITEM id=dfa810df58f9be9f ===

=== ITEM id=b9c701ae1cdb7aa6 ===
Gold sentences and facts:
Sentence 0: The Great War is a 26-episode documentary series from 1964 on the First World War.
  Fact 0.0: The Great War is a documentary series.
  Fact 0.1: The Great War has 26 episodes.
  Fact 0.2: The Great War is a documentary series from 1964.
  Fact 0.3: The Great War is about the First World War.
Sentence 1:  The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian Broadcasting Commission.
  Fact 1.0: The Great War was a co-production of the Imperial War Museum.
  Fact 1.1: The Great War was a co-production of the British Broadcasting Corporation.
  Fact 1.2: The Great War was a co-production of the Canadian Broadcasting Corporation.
  Fact 1.3: The Great War was a co-production of the Australian Broadcasting Commission.
Sentence 2:  The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.
  Fact 2.0: Michael Redgrave was the narrator of The Great War.
  Fact 2.1: Marius Goring gave readings in The Great War.
  Fact 2.2: Ralph Richardson gave readings in The Great War.
  Fact 2.3: Cyril Luckham gave readings in The Great War.
  Fact 2.4: Sebastian Shaw gave readings in The Great War.
  Fact 2.5: Emlyn Williams gave readings in The Great War.
Sentence 3: Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.
  Fact 3.0: Imperial War Museums (IWM) is a British national museum organisation.
  Fact 3.1: Imperial War Museums has branches at five locations in England.
  Fact 3.2: Three of the five Imperial War Museums branches are in London.

Retrieved items:
<item 1>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 1>

<item 2>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 2>

<item 3>
Vital Signs (TV series)
Vital Signs is a British television drama series airing on ITV from 2006.

 It stars Tamzin Outhwaite as a supermarket check-out operator who decides to become a doctor.

 The series co-stars William Beck, Fraser Ayres, Eve Best, Claudie Blakley, Lucinda Dryzek, Beth Goddard, Alfie Hunter, Brooke Kinsella, Harry Lloyd, Peter Rnic
and Steven Waddington.

 The filming of the show is based in numerous London hospitals and medical schools; predominantly the show has been shot in St George's Hospital and Medical
School.
</item 3>

<item 4>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 4>

<item 5>
Paradox (TV series)
Paradox is a 2009 British science fiction police drama, starring Tamzin Outhwaite as Detective Inspector Rebecca Flint.

 Written by Lizzie Mickery and produced by Clerkenwell Films for the BBC, it was filmed and set in Manchester, England.
</item 5>

<item 6>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 6>

<item 7>
First Life (TV series)
First Life is a 2010 British nature documentary series written and presented by David Attenborough, also known by the expanded titles David Attenborough's First
Life (UK) and First Life with David Attenborough (USA).

 It was first broadcast in the USA as a two-hour special on the Discovery Channel on 24 October 2010.

 In the United Kingdom it was broadcast as a two-part series on BBC Two on 5 November 2010.

 "First Life" sees Attenborough tackle the subject of the origin of life on Earth.

 He investigates the evidence from the earliest fossils, which suggest that complex animals first appeared in the oceans around 540 million years ago, an event
known as the Cambrian Explosion.

 Trace fossils of multicellular organisms from an even earlier period, the Ediacaran biota, are also examined.

 Attenborough travels to Canada, Morocco and Australia, using some of the latest fossil discoveries and their nearest equivalents amongst living species to
reveal what life may have been like at that time.

 Visual effects and computer animation are used to reconstruct and animate the extinct life forms.
</item 7>

<item 8>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 8>

<item 9>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 9>

<item 10>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 10>
=== END ITEM id=b9c701ae1cdb7aa6 ===

=== ITEM id=0a968f84a181a365 ===
Gold sentences and facts:
Sentence 0: The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of the Stones River Campaign in the Western Theater of the American Civil War.
  Fact 0.0: The Battle of Stones River is also called the Second Battle of Murfreesboro.
  Fact 0.1: The Battle of Stones River was fought from December 31, 1862, to January 2, 1863.
  Fact 0.2: The Battle of Stones River was fought in Middle Tennessee.
  Fact 0.3: The Battle of Stones River was the culmination of the Stones River Campaign.
  Fact 0.4: The Stones River Campaign took place in the Western Theater of the American Civil War.
Sentence 1: The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.
  Fact 1.0: The Battle of the Ch'ongch'on River is also known as the Battle of the Ch'ongch'on.
  Fact 1.1: The Battle of the Ch'ongch'on River is also known as the Second Phase Campaign Western Sector.
  Fact 1.2: The Battle of the Ch'ongch'on River was a decisive battle in the Korean War.
  Fact 1.3: The Battle of the Ch'ongch'on River took place from November 25 to December 2, 1950.
  Fact 1.4: The Battle of the Ch'ongch'on River took place along the Ch'ongch'on River Valley.
  Fact 1.5: The Ch'ongch'on River Valley is in the northwestern part of North Korea.

Retrieved items:
<item 1>
Battle of Cross Keys
The Battle of Cross Keys was fought on June 8, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Together, the battles of Cross Keys and Port Republic the following day were the decisive victories in Jackson's Valley Campaign, forcing the Union armies to
retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 1>

<item 2>
Mendocino War
The Mendocino War was a violent conflict from July 1859 to January 18, 1860, between white settlers and local natives (mainly Yuki tribes) in Mendocino County,
California.

 It was caused by settler intrusion and slave raids on native lands and subsequent native retaliation, resulting in the deaths of hundreds of natives.

 In 1859, a band of locally sponsored rangers led by Walter S. Jarboe, called the Eel River Rangers, raided the countryside in an effort to remove natives from
settler territory and move them onto the Nome Cult Farm, an area near the Mendocino Indian Reservation.

 By the time the Eel River Rangers were disbanded in 1860, Jarboe and his men had killed 283 warriors, captured 292, killed countless women and children, and
only suffered 5 casualties themselves in just 23 engagements.

 The bill to the state for the rangers’ services amounted to $11,143.43.

 Scholars, however, state that the damage to the area and natives in particular was even higher than reported, especially given the vast number of raiding
parties formed outside of the Eel River Rangers.
</item 2>

<item 3>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 3>

<item 4>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 4>

<item 5>
Joseph L. Follett
Joseph Leonard Follett (February 16, 1843 – April 1, 1907) was an American soldier who fought in the American Civil War.

 Follett received the country's highest award for bravery during combat, the Medal of Honor, for his action during the Battle of Island Number Ten in New
Madrid, Missouri on 3 March 1862 and the Battle of Stones River in Tennessee on 31 December 1862.

 He was honored with the award on 19 September 1890.
</item 5>

<item 6>
Second Battle of Fort Wagner
The Second Battle of Fort Wagner, also known as the Second Assault on Morris Island or the Battle of Fort Wagner, Morris Island, was fought on July 18, 1863,
during the American Civil War.

 Union Army troops commanded by Brig. Gen. Quincy Gillmore, launched an unsuccessful assault on the Confederate fortress of Fort Wagner, which protected Morris
Island, south of Charleston Harbor.

 The battle came one week after the First Battle of Fort Wagner.
</item 6>

<item 7>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 7>

<item 8>
Battle of Port Republic
The Battle of Port Republic was fought on June 9, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Port Republic was a fierce contest between two equally determined foes and was the most costly battle fought by Jackson's Army of the Valley during its
campaign.

 Together, the battles of Cross Keys (fought the previous day) and Port Republic were the decisive victories in Jackson's Valley Campaign, forcing the Union
armies to retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 8>

<item 9>
Battle of Stones River
The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of
the Stones River Campaign in the Western Theater of the American Civil War.

 Of the major battles of the Civil War, Stones River had the highest percentage of casualties on both sides.

 Although the battle itself was inconclusive, the Union Army's repulse of two Confederate attacks and the subsequent Confederate withdrawal were a much-needed
boost to Union morale after the defeat at the Battle of Fredericksburg, and it dashed Confederate aspirations for control of Middle Tennessee.
</item 9>

<item 10>
Roscoe Channing
Roscoe H. Channing, Jr. (January 7, 1868 – April 1, 1961) was an All-American football player, member of the Rough Riders and mining executive.

 Channing was an All-American halfback for Princeton University.

 He was one of eleven players selected by Caspar Whitney for the first ever College Football All-America Team in 1889.

 When the Spanish–American War commenced in 1898, Channing enlisted in Theodore Roosevelt's Rough Riders.

 Roosevelt took pride in how many Ivy League football players enlisted in the Rough Riders.

 Channing later went into the mining business and managed the mining operations of the Whitney family.

 In the 1920s, he formed a partnership with his friend Cornelius Vanderbilt Whitney.

 The two formed the Hudson Bay Mining and Smelting Company in Flin Flon, Saskatchewan, Canada, and Channing served as the company's President.

 Channing died in 1961.
</item 10>
=== END ITEM id=0a968f84a181a365 ===

=== ITEM id=62017487881c57bc ===
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
Nancy Schreiber
Nancy Schreiber (born June 27, 1949) is an American cinematographer known for her work on "Chain of Desire", "Dead Beat", "The Celluloid Closet", "November",
and "The Nines".
</item 1>

<item 2>
Overnight Delivery
Overnight Delivery is a 1998 romantic comedy film directed by Jason Bloom.

 It was rated PG-13 by the MPAA and released direct-to-video.

 It featured Reese Witherspoon and Paul Rudd, prior to both becoming considerably bigger film stars.
</item 2>

<item 3>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 3>

<item 4>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 4>

<item 5>
Reese Witherspoon
Laura Jeanne Reese Witherspoon ( ; born March 22, 1976) is an American actress, producer, and entrepreneur.

 Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for
which she was nominated for a Young Artist Award.

 Following breakout roles in "" (1992) and "Jack the Bear" (1993), she starred in the comedy-drama "Pleasantville" (1998), for which she won the Young Hollywood
Award for Breakthrough Performance.

 Her leading role of Tracy Flick in "Election" (1999) was nominated for a Golden Globe Award.
</item 5>

<item 6>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 6>

<item 7>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 7>

<item 8>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 8>

<item 9>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 9>

<item 10>
City of New Orleans (train)
The City of New Orleans is an Amtrak passenger train which operates on an overnight schedule between Chicago and New Orleans.

 The train is a successor to the Illinois Central Railroad's "Panama Limited".

 The present name was revived in 1981.
</item 10>
=== END ITEM id=62017487881c57bc ===

=== ITEM id=b5993490c1d7e2cd ===
Gold sentences and facts:
Sentence 0: The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
  Fact 0.0: The International Grenfell Association (IGA) is an organization.
  Fact 0.1: The International Grenfell Association was founded by Sir Wilfred Grenfell.
  Fact 0.2: The International Grenfell Association provides health care.
  Fact 0.3: The International Grenfell Association provides education.
  Fact 0.4: The International Grenfell Association provides religious services.
  Fact 0.5: The International Grenfell Association provides rehabilitation.
  Fact 0.6: The International Grenfell Association provides other social activities.
  Fact 0.7: The International Grenfell Association serves fisherman and coastal communities in northern Newfoundland.
  Fact 0.8: The International Grenfell Association serves fisherman and coastal communities on the coast of Labrador.
Sentence 1: Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
  Fact 1.0: Sir Wilfred Thomason Grenfell held the title KCMG.
  Fact 1.1: Sir Wilfred Thomason Grenfell was born on 28 February 1865.
  Fact 1.2: Sir Wilfred Thomason Grenfell died on 9 October 1940.
  Fact 1.3: Sir Wilfred Thomason Grenfell was a medical missionary to Newfoundland.
  Fact 1.4: Sir Wilfred Thomason Grenfell was a medical missionary to Labrador.

Retrieved items:
<item 1>
Wilfred Grenfell
Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
</item 1>

<item 2>
Church of St. Ignatius Loyola (New York City)
The Church of St. Ignatius of Loyola is a Roman Catholic parish church located on the Upper East Side of Manhattan, New York City, administered by the Society
of Jesus (Jesuits).

 The parish is under the authority of the Archdiocese of New York, and was established in 1851 as St. Lawrence O'Toole's Church.

 In 1898, permission to change the patron saint of the parish from St. Lawrence O’Toole to St. Ignatius of Loyola was granted by Rome.

 The address is 980 Park Avenue, New York City, New York 10028.

 The church on the southwest corner of Park Avenue and 84th Street is part of a Jesuit complex on the block that includes Wallace Hall, the parish hall, beneath
the church, the rectory at the midblock location on Park Avenue, the grade school of St.
</item 2>

<item 3>
Ellis Gowing
Gowing was educated at Fort Street High School and the University of Sydney; and ordained in 1907.

 He began his career as a Curate at The Oaks, New South Wales, after which he served at Armidale Cathedral.

 Coming to England he was on the staff of St James the Less, Bethnal Green then domestic chaplain to the Bishop of Chelmsford from 1914 to 1917.

 He was vicar of Prittlewell from 1917 until his death; an honorary canon of Chelmsford Cathedral from 1921 to 1938 and Rural Dean of Canewdon from 1918 to
1938.

 He was given the Freedom of the County Borough of Southend-on-Sea in 1953.
</item 3>

<item 4>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 4>

<item 5>
International Grenfell Association
The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and
rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
</item 5>

<item 6>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 6>

<item 7>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 7>

<item 8>
Martha Wall
Martha Alma Wall (March 22, 1910 – August 2, 2000) was an American Christian medical missionary, philosopher, nurse, and author who is best known for her
humanitarian work providing health care to lepers in British Nigeria during the 1930s and 1940s with the Sudan Interior Mission (SIM).

 She was born in Hillsboro, Kansas to a traditional Christian family and was a devout member of both the non-denominational Salina Bible Church and the Baptist
Women's Union.

 She became a registered nurse and studied theology at Tabor College before leaving for a medical mission in British Nigeria in 1938.

 After returning to America, Wall worked as a Clinical Supervisor of Vocational Nurses for Kern General Hospital during the 1950s and as an instructor and
director of nursing services for Bakersfield College during the 1960s.

 Throughout her adult life, she was a dedicated member of the California State Licensed Vocational Nurses Association.

 Wall is noted as founder of the Children's Welfare Center at the Katsina Leper Settlement.

 She documented her missionary work in Sub-Saharan Africa in the book she authored "Splinters from an African Log", which was published in 1960.
</item 8>

<item 9>
Paul Schulte
Paul Schulte OMI (1896 – 1975), was a German priest and missionary, known as the "Flying Priest", who founded MIVA ("Missionary International Vehicular
Association") to provide automobiles, boats and airplanes for the service of missions throughout the world.
</item 9>

<item 10>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 10>
=== END ITEM id=b5993490c1d7e2cd ===

=== ITEM id=61d1216f1e971c1f ===
Gold sentences and facts:
Sentence 0:  Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
  Fact 0.0: Her personal best throw is 58.87 metres.
  Fact 0.1: She achieved her personal best throw at the 2002 Asian Games.
  Fact 0.2: The 2002 Asian Games were held in Busan.
Sentence 1:  A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.
  Fact 1.0: A total of 460 athletes took part in the competition.
  Fact 1.1: The athletes comprised 281 men and 179 women.
  Fact 1.2: The athletes came from 39 nations.

Retrieved items:
<item 1>
Daniel Ståhl
Daniel Ståhl (born 27 August 1992) is a Swedish athlete specialising in the discus throw.

 He competed at the 2015 World Championships in Beijing finishing fifth.

 In 2016, he competed at the European Championships, where he finished fifth.

 Daniel also competed at the Olympic Games in Rio de Janeiro the same year, where he failed to qualify for the final.

 Just a couple of weeks following his failure in Rio, Daniel competed at the Swedish Championships in Sollentuna.

 Not only did he win the discus throw on a new personal best 68.72 metres, it was also the new world leading throw of 2016, surpassing Christoph Harting's 68.37
metres from the Rio Olympics final.

 By the end of the year, it remained the number one throw in 2016.

 The day before, Daniel had also won the shot put on a new personal best 19.38 metres.
</item 1>

<item 2>
Lee Young-sun
Lee Young-Sun (born 21 February 1974) is a retired female javelin thrower from South Korea.

 Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
</item 2>

<item 3>
Zhao Qinggang
Zhao Qinggang (; born 24 July 1985) is a Chinese track and field athlete who competes in the javelin throw.

 His 89.15 m personal best throw, set in 2014, is the Chinese record.

 It was also the Asian record until surpassed in 2017.

 He represented his country at the 2013 World Championships in Athletics.

 He is the 2014 Asian Games champion.

 At the East Asian Games he was the runner-up in 2009 and winner in 2013.

 He is a two-time Chinese champion (2012 and 2013) and was the winner of the 2013 National Games of China.
</item 3>

<item 4>
2009 Hansol Korea Open – Doubles
Chuang Chia-jung and Hsieh Su-wei were the defending champion, but Hsieh chose not to participate that year.

 Chuang partnered with Yan Zi, but they lost in the semifinals against Chan Yung-jan and Abigail Spears.

Chan Yung-jan and Abigail Spears won in the final 6–3, 6–4 against Carly Gullickson and Nicole Kriz.
</item 4>

<item 5>
Masayoshi Yamamoto
Masayoshi Yamamoto (Japanese: 山本 雅賢) is a Japanese male artistic gymnast, representing his nation at international competitions.

 He won the gold medal in the Pommel horse event at the 2014 Asian Games in Incheon, South Korea.
</item 5>

<item 6>
Lee–Lin rivalry
They have contested in many major tournament finals and are currently the only two badminton singles players who have contested in two Olympic finals where Lin
Dan won both of the time.

 They have also contested two BWF World Championships finals and one Asian Games final where all matches were very close hard fought 3 setters and won by Lin
Dan.

 There 2011 World Badminton Championship match widely touted as one of the greatest badminton matches of all the time.

 In the semi-finals of the 2016 Rio Olympics, most possibly the last Olympic Games of both athletes' careers, Lee Chong Wei won against Lin Dan, for the first
time on the Olympic stage, making Lee the second shuttler in any discipline of the sport in history to reach 3 consecutive finals after Fu Haifeng who had
achieved the feat a few hours earlier in the men doubles.

 The two players exchanged a special moment in their rivalry and exchanged shirts as they exited the court.

 The 2016 Rio Games marks an end in the badminton era dominated by Lee and Lin.
</item 6>

<item 7>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 7>

<item 8>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 8>

<item 9>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 9>

<item 10>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 10>
=== END ITEM id=61d1216f1e971c1f ===
