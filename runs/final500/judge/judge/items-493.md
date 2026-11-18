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

=== ITEM id=f642b7e90d3fd35b ===
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
Standing at the Sky's Edge
Standing at the Sky's Edge is the seventh studio album from English musician Richard Hawley, released in the UK on 7 May 2012 and in the US on 28 August 2012.

 The album is markedly different from Hawley's previous efforts, often relinquishing softer instrumentation in favour of squalling guitars.

 As with all of Hawley's previous albums, the title obliquely refers to a location in his native city of Sheffield, in this case Skye Edge, a hillside area with
views over the city centre and formerly known for its crime-ridden estates but largely redeveloped in the mid-2000s.
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
Range Life (song)
The single was not commercially released in the USA; it was issued by the band's UK label at the time, Big Cat.

 Both B-sides are outtakes from the "Crooked Rain, Crooked Rain" sessions and are included on the 2004 of that album.

 This song was one of many to be included in the group's greatest hits album "".
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
Beatles (song)
"Beatles", written by Sven Olov Bagge and Claes Bure, was the song that, performed by Swedish dansband Forbes, won the Swedish Melodifestivalen 1977.

 The song's lyrics are about the popular 1960s British rock group the Beatles and include the phrase "Yeah, yeah, yeah" as an homage to the Beatles song "She
Loves You".

 Forbes took the song to the United Kingdom for the Eurovision Song Contest 1977, with Anders Berglund as conductor.

 The audience cheered at the song, not unexpectedly since the contest took place in the United Kingdom, but the jury had another opinion.

 The song finished last, with only two points, from West Germany.
</item 9>

<item 10>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 10>
=== END ITEM id=f642b7e90d3fd35b ===

=== ITEM id=ca0803d034736231 ===
Gold sentences and facts:
Sentence 0:  The trains were built between 1994 and 2009 – for the most part by Talbot, part of Bombardier Inc., with some railcars built by De Dietrich.
  Fact 0.0: The trains were built between 1994 and 2009.
  Fact 0.1: The trains were built for the most part by Talbot.
  Fact 0.2: Talbot is part of Bombardier Inc.
  Fact 0.3: Some railcars were built by De Dietrich.
Sentence 1: Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.
  Fact 1.0: Bombardier Inc. is a multinational aerospace and transportation company.
  Fact 1.1: Bombardier Inc. is based in Montreal, Quebec, Canada.

Retrieved items:
<item 1>
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 1>

<item 2>
Northern Transportation Company
Northern Transportation Company Limited (NTCL) is a marine transportation company in the Canadian and American Arctic owned by Norterra, a holding company that
is owned by the Inuvialuit of the Northwest Territories.

 While primarily a marine freight hauler on Canadian rivers, like the Mackenzie and the Hay and along the Arctic coast of Canada, they are also a petroleum
wholesaler.

 Its head office is now in the town of Hay River, Northwest Territories.

 Today, it primarily utilizes tugs and barges.
</item 2>

<item 3>
Graph query row:
transportation_company: Bombardier Inc.
based_in: Montreal
relationship: PART_OF
</item 3>

<item 4>
Toland, Pennsylvania
Toland is an unincorporated community in Cumberland County, Pennsylvania, United States.

 The very small village is located on Pine Grove Road, east of Mountain Creek Campground.

 Michaux State Forest, the site of Laurel Lake, Fuller Lake and Pine Grove Furnace State Park lie a few miles to the west near the intersection of Pine Grove
Road and PA route 233.

 Toland has a mailing address of Gardners Pennsylvania, because the size of the community doesn't warrant a post-office, nor an official incorporated name.

 What is now a sand pit operation was originally a clay bank mining operation that was the reason for the location of the village of Toland.

 Less than 50 people reside in the close-knit community, which was built for the clay bank company workers in the first quarter of the 20th century.

 The original community of Toland, Pennsylvania consisted of 11 duplex houses, built side by side along Pine Grove Road, with less than 0.17 of an acre of land
to each.

 Additional homes have been built since.

 When the community was originally built there was only a common well with a hand pump for all of the families to draw from.
</item 4>

<item 5>
Marquette Transportation Company
The Marquette Transportation Company is a marine transportation company based in Paducah, Kentucky, United States.
</item 5>

<item 6>
Merchants Transportation Company
The Merchants Transportation Company was a shipping firm that operated on Puget Sound from 1905 to 1929.

 This company should not be confused with the similarly named Merchants Transportation Company of Olympia, formed in 1874 and a completely separate firm.
</item 6>

<item 7>
Kitsap County Transportation Company
The Kitsap County Transportation Company was an important steamboat and ferry company that operated on Puget Sound.

 The company was founded in 1898 as the Hansen Transportation Company.
</item 7>

<item 8>
Volvo Halifax Assembly
The Volvo Halifax Assembly Plant located in Halifax, Nova Scotia was opened on 11 June 1963 by Prince Bertil.

 It was the first assembly plant Volvo opened outside of Sweden and the first non-domestic auto plant in North America.

 Volvo decided to open to the plant to bypass hefty North American import tariffs on foreign goods and to capitalize on the newly signed Canadian/American Auto
Pact.

 The plant was operated by Volvo Canada Limited (now Volvo Cars of Canada Corporation) in Toronto, Ontario and bridged the gap between Volvo of North America
(Rockleigh, New Jersey), Volvo headquarters and the flagship Torslanda plant in Gothenburg.
</item 8>

<item 9>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 9>

<item 10>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 10>
=== END ITEM id=ca0803d034736231 ===

=== ITEM id=aeadb475a8697fef ===
Gold sentences and facts:
Sentence 0: Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.
  Fact 0.0: Georg Ferdinand Ludwig Philipp Cantor was born on March 3, 1845.
  Fact 0.1: The birth date of Georg Ferdinand Ludwig Philipp Cantor is also given as February 19, 1845 in the Old Style calendar.
  Fact 0.2: Georg Ferdinand Ludwig Philipp Cantor died on January 6, 1918.
  Fact 0.3: Georg Ferdinand Ludwig Philipp Cantor was a German mathematician.
Sentence 1: Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian mathematician.
  Fact 1.0: Vladimir Igorevich Arnold is also spelled Arnol'd.
  Fact 1.1: Vladimir Igorevich Arnold's name in Russian is Влади́мир И́горевич Арно́льд.
  Fact 1.2: Vladimir Igorevich Arnold was born on 12 June 1937.
  Fact 1.3: Vladimir Igorevich Arnold died on 3 June 2010.
  Fact 1.4: Vladimir Igorevich Arnold was a Soviet mathematician.
  Fact 1.5: Vladimir Igorevich Arnold was a Russian mathematician.

Retrieved items:
<item 1>
Jónsson–Tarski algebra
In mathematics, a Jónsson–Tarski algebra or Cantor algebra is an algebraic structure encoding a bijection from an infinite set "X" onto the product "X"×"X".

 They were introduced by .

 , named them after Georg Cantor because of Cantor's pairing function and Cantor's theorem that an infinite set "X" has the same number of elements as "X"×"X";
the term "Cantor algebra" is also occasionally used to mean the Boolean algebra of all clopen subsets of the Cantor set, or the Boolean algebra of Borel subsets
of the reals modulo meager sets (sometimes called the Cohen algebra).
</item 1>

<item 2>
Alfred von Arneth
Alfred Ritter von Arneth (10 July 181930 July 1897) was an Austrian historian.

 His principal scholarly work is a ten-volume biography of the Habsburg Empress Maria Theresa, first published in installments from 1863 to 1879 and still
regarded as the standard work on the subject.

 Born at Vienna, he was the son of Joseph Calasanza von Arneth (1791–1863), a well-known historian and archaeologist, who wrote a history of the Austrian Empire
(Vienna, 1827) and several works on numismatics and brother of Doctor Franz Hektor von Arneth (1818–1907).
</item 2>

<item 3>
Georg Cantor
Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.

 He invented set theory, which has become a fundamental theory in mathematics.

 Cantor established the importance of one-to-one correspondence between the members of two sets, defined infinite and well-ordered sets, and proved that the
real numbers are more numerous than the natural numbers.

 In fact, Cantor's method of proof of this theorem implies the existence of an "infinity of infinities".

 He defined the cardinal and ordinal numbers and their arithmetic.

 Cantor's work is of great philosophical interest, a fact of which he was well aware.
</item 3>

<item 4>
Cantor–Bernstein theorem
In set theory and order theory, the Cantor–Bernstein theorem states that the cardinality of the second type class, the class of countable order types, equals
the cardinality of the continuum.

 It was used by Felix Hausdorff and named by him after Georg Cantor and Felix Bernstein.

 Cantor constructed a family of countable order types with the cardinality of the continuum, and in his 1901 inaugural dissertation Bernstein proved that such a
family can have no higher cardinality.
</item 4>

<item 5>
Anton Schwarzkopf
Anton Schwarzkopf (8 July 1924 – 30 July 2001) was a German engineer of amusement rides, and founder of the Schwarzkopf Industries Company, which built numerous
amusement rides and large roller coasters for both amusement parks and traveling funfairs.
</item 5>

<item 6>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 6>

<item 7>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 7>

<item 8>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 8>

<item 9>
Cantor space
In mathematics, a Cantor space, named for Georg Cantor, is a topological abstraction of the classical Cantor set: a topological space is a Cantor space if it is
homeomorphic to the Cantor set.

 In set theory, the topological space 2 is called "the" Cantor space.

 Note that, commonly, 2 is referred to simply as the Cantor set, while the term Cantor space is reserved for the more general construction of "D" for a finite
set "D" and a set "S" which might be finite, countable or possibly uncountable.
</item 9>

<item 10>
Georg Cantor's first set theory article
Georg Cantor's first set theory article was published in 1874 and contains the first theorems of transfinite set theory, which studies infinite sets and their
properties.

 One of these theorems is "Cantor's revolutionary discovery" that the set of all real numbers is uncountably, rather than countably, infinite.

 This theorem is proved using Cantor's first uncountability proof, which differs from the more familiar proof using his diagonal argument.

 The title of the article, "On a Property of the Collection of All Real Algebraic Numbers" ("Ueber eine Eigenschaft des Inbegriffes aller reellen algebraischen
Zahlen "), refers to its first theorem: the set of real algebraic numbers is countable.
</item 10>
=== END ITEM id=aeadb475a8697fef ===

=== ITEM id=2b1f3807580cc8c3 ===
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
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 1>

<item 2>
Gentleman Jole and the Red Queen
Gentleman Jole and the Red Queen is a science fiction novel by Lois McMaster Bujold.

 It is an installment in Bujold's Vorkosigan Saga.

 Bujold has described it as "not a war story.

 It is about grownups."
</item 2>

<item 3>
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
</item 3>

<item 4>
Passage (Bujold novel)
Passage is a novel by American writer Lois McMaster Bujold, published in 2008.

 It is the third in the tetralogy "The Sharing Knife".
</item 4>

<item 5>
Shards of Honor
Shards of Honor is an English language science fiction novel by Lois McMaster Bujold, first published in June 1986.

 It is a part of the Vorkosigan Saga, and is the first full-length novel in publication order.

 "Shards of Honor" is paired with Bujold's 1991 "Barrayar" in the omnibus "Cordelia's Honor" (1996).
</item 5>

<item 6>
Graph query row:
person: Bernard Malamud
birth_date: April 26, 1914
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
Cryoburn
Cryoburn is a science fiction novel by Lois McMaster Bujold, first published in October 2010.

 Part of the Vorkosigan Saga, it was nominated for the Hugo Award for Best Novel in 2011, as Bujold's ninth Best Novel nomination.

 Also in 2011, it was one of the top five finishers in the poll for the Locus Award for Best Science Fiction Novel.
</item 8>

<item 9>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 9>

<item 10>
Graph query row:
person: Lois McMaster Bujold
birth_date: November 2, 1949
</item 10>
=== END ITEM id=2b1f3807580cc8c3 ===

=== ITEM id=9f883fc181ca8c4d ===
Gold sentences and facts:
Sentence 0: The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.
  Fact 0.0: The Bears and I is a 1974 American drama film.
  Fact 0.1: The Bears and I was directed by Bernard McEveety.
  Fact 0.2: The Bears and I was written by John Whedon.
Sentence 1:  The film was released on July 31, 1974, by Buena Vista Distribution.
  Fact 1.0: The Bears and I was released on July 31, 1974.
  Fact 1.1: The Bears and I was released by Buena Vista Distribution.
Sentence 2: Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as co-director.
  Fact 2.0: Oceans (French: Océans) is a 2009 French nature documentary film.
  Fact 2.1: Oceans was directed by Jacques Perrin.
  Fact 2.2: Oceans was produced by Jacques Perrin.
  Fact 2.3: Oceans was co-written by Jacques Perrin.
  Fact 2.4: Oceans was narrated by Jacques Perrin.
  Fact 2.5: Jacques Cluzaud was co-director of Oceans.

Retrieved items:
<item 1>
Return to the Blue Lagoon
Return to the Blue Lagoon is a 1991 American romance and adventure film starring Milla Jovovich and Brian Krause, produced and directed by William A. Graham.

 The film is a sequel to "The Blue Lagoon" (1980).

 The screenplay by Leslie Stevens was based on the novel "The Garden of God" by Henry De Vere Stacpoole.

 The original music score was composed by Basil Poledouris.

 The film's closing theme song "A World of Our Own" is performed by Surface featuring Bernard Jackson.

 The music was written by Barry Mann, and the lyrics were written by Cynthia Weil.
</item 1>

<item 2>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 2>

<item 3>
Foxes (soundtrack)
Foxes is the soundtrack to the 1980 film of the same name, starring Jodie Foster, Scott Baio, Sally Kellerman, Randy Quaid as well as The Runaways' lead singer
Cherie Currie.

 The double-album was released on the disco label Casablanca Records.
</item 3>

<item 4>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 4>

<item 5>
The Best of Creedence Clearwater Revival
The Best of Creedence Clearwater Revival is a compilation album by American rock band Creedence Clearwater Revival, released in 1977.

 The album features all the tracks of Chronicle which was released a year earlier in 1976, as well as "Good Golly Miss Molly", "Born on the Bayou", "Cotton
Fields", "Hello Mary Lou", "The Midnight Special", "Walk on the Water", and "Night Time Is the Right Time" which would appear on the 1986 release "Chronicle,
Vol.

 2", and "Bootleg" which does not appear on either volume.
</item 5>

<item 6>
I Against I
I Against I is the third studio album by the American hardcore punk band Bad Brains.

 It was released in November 1986 through SST Records with the catalog number SST 065.

 The best-selling album in the band's catalog, "I Against I" is a critically acclaimed album that mixes American hardcore punk with funk, soul, reggae and heavy
metal.

 It is also included in the book "1001 Albums You Must Hear Before You Die".

 The album featured an MTV video for the title track directed by Paul Rachman who later went on to direct the documentary feature film "American Hardcore".
</item 6>

<item 7>
Buena Vista (brand)
Buena Vista is a brand name which was historically often used for divisions and subsidiaries of The Walt Disney Company, whose primary studios, the Walt Disney
Studios, are located on Buena Vista Street in Burbank, California.

 (The studio lot is also home to the Company's corporate headquarters, the Team Disney Burbank building.)

 The logos for the various Buena Vista brands featured the "Buena Vista" wordmark superimposed over the Disney Castle image to signal the affiliation between
Buena Vista and Disney.

 The name literally means "good view" in Spanish.
</item 7>

<item 8>
Entre el Cielo y el Infierno
Entre el Cielo y el Infierno (Between Heaven and Hell) is the fifth studio album by Argentine heavy metal band Rata Blanca.

 The album was recorded at Estudios La Nave between June and July ,1994 and was released in September, 1994 by BMG.

 Is the first and unique album featuring vocalist Mario Ian and also is the first album featuring Javier Retamozo on keyboards.
</item 8>

<item 9>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 9>

<item 10>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 10>
=== END ITEM id=9f883fc181ca8c4d ===

=== ITEM id=fba80f182b5c4f10 ===
Gold sentences and facts:
Sentence 0: Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.
  Fact 0.0: Catfish and the Bottlemen are a British rock band.
  Fact 0.1: Catfish and the Bottlemen were formed in Llandudno, Wales.
  Fact 0.2: Catfish and the Bottlemen were formed in 2007.
Sentence 1: The Lumineers are an American folk rock/Americana band based in Denver, Colorado.
  Fact 1.0: The Lumineers are an American folk rock/Americana band.
  Fact 1.1: The Lumineers are based in Denver, Colorado.
Sentence 2:  Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.
  Fact 2.0: Schultz and Fraites began writing together in 2005.
  Fact 2.1: Schultz and Fraites began performing together in 2005.
  Fact 2.2: Schultz and Fraites began writing together in Ramsey, New Jersey.
  Fact 2.3: Schultz and Fraites began performing together in Ramsey, New Jersey.
Sentence 3:  The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.
  Fact 3.0: The Lumineers emerged as one of the most popular folk rock/Americana artists.
  Fact 3.1: The Lumineers emerged during the revival of folk rock and Americana genres.
  Fact 3.2: The Lumineers emerged during the growing popularity of folk rock and Americana in the 2010s.

Retrieved items:
<item 1>
The Ride (Catfish and the Bottlemen album)
The Ride is the second studio album by British indie rock band Catfish and the Bottlemen.

 It was released on 27 May 2016 through Capitol Records.
</item 1>

<item 2>
The Ride Tour
The Ride Tour is the ongoing second concert tour by English indie rock band Catfish and the Bottlemen, in support of the group's second studio album "The Ride".

 The tour was announced in February 2016 and began on 4 April 2016, in Glasgow, United Kingdom at the O2 Academy Glasgow.

 The tour is planned to travel across Europe and North America with a total of 50 shows planned so far.

 On 22 February 2016, the band announced an April 2016 stint of shows in UK, including Glasgow, Halifax, Southend On Sea, Doncaster, London and Brighton.

 The band will also headline several festivals across the UK throughout the summer, including Liverpool's 'Sound City' in May 2016, Bristol's Summer Series in
July 2016 and Manchester's Castlebowl in July 2016.

 On 14 March 2016 it was announced that the band will play several dates in North America in June 2016.
</item 2>

<item 3>
Billy Bibby &amp; The Wry Smiles
Billy Bibby & The Wry Smiles are a British indie rock band formed in the autumn of 2015 in the area around Llandudno, North Wales and Chester, UK.

 The band is fronted by Billy Bibby, former lead guitarist and founding member of Catfish and the Bottlemen.

 Bibby parted ways with Catfish and the Bottlemen in the summer of 2014, shortly after co-writing and recording their Platinum selling debut album, "The
Balcony".

 Soon thereafter, Billy Bibby began writing songs and playing acoustic shows in various UK cities in order to test public response to his new music.

 In the spring of 2015, Billy Bibby began seeking musicians to form a new band.

 Auditions were conducted with the help of Simon Jones, formerly of The Verve.

 Band members Rob Jones (lead guitar), Matt Bassy (bass), and Michael Pearce (drums) were selected to form the four-piece ensemble.
</item 3>

<item 4>
The Balcony (album)
The Balcony is the debut studio album by Welsh rock band Catfish and the Bottlemen.

 It was released on 15 September 2014 in the United Kingdom and 15 January 2015 in the United States.

 On 20 March 2015, "The Balcony" was awarded a Gold certification from the British Phonographic Industry.

 The album cover depicts outlines of two headless people pleasuring each other's genitals.

 The illustration is by New York artist Tim Lahan who originally posted it to his Flickr account in 2009 and was subsequently contacted by the band.
</item 4>

<item 5>
Ryan’s Gig Guide covers Birmingham and the Black Country music scene. The Preytells were a Perth indie-rock band (2004–09), releasing one album and earning WAM
Song of the Year nominations. English pop-rockers the Beautiful South formed in 1988 with former Housemartins Paul Heaton and Dave Hemingway. Welsh band Catfish
and the Bottlemen released *The Balcony* (2014) and *The Ride* (2016); their 50-date
</item 5>

<item 6>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 6>

<item 7>
Kathleen (song)
"Kathleen" is the third single by Welsh indie rock band, Catfish and the Bottlemen.

 The song was included in their EP, "Kathleen and the Other Three", and their debut studio album, "The Balcony".

 The single was digitally released on 18 March 2014 as a music video, and on 7 April 2014, the 7-inch version of the single was released.

 The single did not contain a B-side.
</item 7>

<item 8>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 8>

<item 9>
Soundcheck (song)
"Soundcheck" is the eighth single by Welsh indie rock band, Catfish and the Bottlemen.

 The song was the lead single off of their sophomore album, "The Ride".

 The single was a promotional single that was digitally released on 16 February 2016.
</item 9>

<item 10>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 10>
=== END ITEM id=fba80f182b5c4f10 ===

=== ITEM id=8e554b921fed5656 ===
Gold sentences and facts:
Sentence 0: Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.
  Fact 0.0: Jean-François André Clervoy was born on 19 November 1958.
  Fact 0.1: Jean-François André Clervoy is a French engineer.
  Fact 0.2: Jean-François André Clervoy is a CNES astronaut.
  Fact 0.3: Jean-François André Clervoy is an ESA astronaut.
Sentence 1: Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
  Fact 1.0: Prof. Dr. Ulrich Hans Walter was born on February 9, 1954.
  Fact 1.1: Prof. Dr. Ulrich Hans Walter is a German physicist.
  Fact 1.2: Prof. Dr. Ulrich Hans Walter is a German engineer.
  Fact 1.3: Prof. Dr. Ulrich Hans Walter is a former DFVLR astronaut.

Retrieved items:
<item 1>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 1>

<item 2>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 2>

<item 3>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 3>

<item 4>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 4>

<item 5>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 5>

<item 6>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 6>

<item 7>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 7>

<item 8>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 8>

<item 9>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 9>

<item 10>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 10>
=== END ITEM id=8e554b921fed5656 ===

=== ITEM id=cd736444545767ec ===
Gold sentences and facts:
Sentence 0: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 0.0: Warrel Dane was born on March 7, 1969.
  Fact 0.1: Warrel Dane is the lead singer of the band Sanctuary.
  Fact 0.2: Sanctuary is an American power metal band.
Sentence 1: Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.
  Fact 1.0: Roy Sætre Khantatat was born on March 12, 1970.
  Fact 1.1: Roy Sætre Khantatat is a Norwegian singer-songwriter.

Retrieved items:
<item 1>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 1>

<item 2>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 2>

<item 3>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 3>

<item 4>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 4>

<item 5>
Roy Khan
Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.

 Commonly known as Roy Khan or simply Khan, he was the lead vocalist for the symphonic power metal band Kamelot from 1997 to his departure in 2011.

 He co-wrote most of Kamelot's songs with the band's guitarist and founder Thomas Youngblood during his tenure.

 Prior to joining Kamelot, Khan was lead singer for the Norwegian progressive metal band Conception from 1991 until 1997 when they disbanded.
</item 5>

<item 6>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 6>

<item 7>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 7>

<item 8>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 8>

<item 9>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 9>

<item 10>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 10>
=== END ITEM id=cd736444545767ec ===

=== ITEM id=8b85154c6b4c834e ===
Gold sentences and facts:
Sentence 0: Fastbacks were a Seattle, Washington, punk rock band.
  Fact 0.0: Fastbacks were a punk rock band.
  Fact 0.1: Fastbacks were from Seattle, Washington.
Sentence 1: Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.
  Fact 1.0: Local Natives is an American indie rock band.
  Fact 1.1: Local Natives was previously known as Cavil at Rest.
  Fact 1.2: Local Natives is based in Silver Lake, Los Angeles, California, United States.

Retrieved items:
<item 1>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 1>

<item 2>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 2>

<item 3>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 3>

<item 4>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 4>

<item 5>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 5>

<item 6>
Fastbacks
Fastbacks were a Seattle, Washington, punk rock band.

 Formed in 1979 by songwriter/guitarist Kurt Bloch (born August 28, 1960), and friends Lulu Gargiulo (guitar and vocals, born October 12, 1960) and Kim Warnick
(bass and vocals, born April 7, 1959), they disbanded in 2001.

 Their sound mixed a generally punk rock approach to vocals and sound textures with poppy tunes and strong musicianship.
</item 6>

<item 7>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 7>

<item 8>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 8>

<item 9>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 9>

<item 10>
Hummingbird (Local Natives album)
Hummingbird is the second studio album by American indie rock band Local Natives, released on January 29, 2013 on Frenchkiss Records in the US, and on
Infectious Records in Europe.

 Produced by The National guitarist Aaron Dessner, the album was preceded by the singles, "Breakers" and "Heavy Feet."

 "Breakers" reached #12 on the Billboard Top 200.

 The song "Mt. Washington" was also featured in the 2015 video game "Life Is Strange".
</item 10>
=== END ITEM id=8b85154c6b4c834e ===

=== ITEM id=f007be43802e4089 ===
Gold sentences and facts:
Sentence 0:  In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.
  Fact 0.0: In 2010, the person referred to as "he" began providing vocals in the pop rock band Ashland HIGH.
  Fact 0.1: In 2010, the person referred to as "he" began providing guitar in the pop rock band Ashland HIGH.
Sentence 1: Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.
  Fact 1.0: Jizzy Pearl was born James Wilkinson.
  Fact 1.1: Jizzy Pearl is an American vocalist.
  Fact 1.2: Jizzy Pearl was born on March 17, 1958.
  Fact 1.3: Jizzy Pearl was born in Chicago, Illinois.

Retrieved items:
<item 1>
Trace Cyrus
Trace Dempsey Cyrus (born Neil Timothy Helson; February 24, 1989) is an American musician.

 The son of country singer Billy Ray Cyrus and brother of recording artist Miley Cyrus, he is the backing vocalist and bassist of the band Metro Station.

 He left the band in 2010 but returned in 2014.

 In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.

 He also owns the clothing company From Backseats to Bedrooms.
</item 1>

<item 2>
Brother Clyde (album)
Brother Clyde is the debut studio album from alternative rock group Brother Clyde.

 It is the first side-project from country music singer Billy Ray Cyrus.

 The album features duets with Cyrus' son Trace Cyrus as well as Dolly Parton, who Cyrus said "wanted to rock."

 It was released on August 10, 2010.

 Its first single, "Lately", which features rapper King Phaze, was issued to digital retailers on June 29, 2010, and a music video, which was directed by Cyrus,
was released for the song in July 2010.
</item 2>

<item 3>
Noah Cyrus
Noah Lindsey Cyrus (born January 8, 2000) is an American actress and singer.

 She voiced the title role in the English version of the 2009 animated feature film "Ponyo".

 In 2016, she released her debut single "Make Me (Cry)", featuring vocals from Labrinth.

 Her debut album is set to be released in late 2017.

 She is the youngest daughter of Tish Cyrus and Billy Ray Cyrus and the younger sister of Miley Cyrus and Trace Cyrus.
</item 3>

<item 4>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 4>

<item 5>
Jizzy Pearl
Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.

 He's had notable performances in several heavy metal and hard rock groups.

 He first fronted the band Data Clan, which eventually became Love/Hate.

 Pearl has also sung for L.A. Guns, Ratt, Adler's Appetite, and other, lesser known acts.
</item 5>

<item 6>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 6>

<item 7>
Graph query row:
person: Jizzy Pearl
occupation: vocalist
relationship: MEMBER_OF
member_of: L.A. Guns
</item 7>

<item 8>
Ron Cyrus
Ronald Ray "Ron" Cyrus (July 10, 1935 – February 28, 2006) was an American Democratic politician and public servant in Greenup County, Kentucky.

 He was the father of American country music singer/actor Billy Ray Cyrus and the paternal grandfather of Trace Cyrus, Miley Cyrus and Noah Cyrus.
</item 8>

<item 9>
Graph query row:
person: Trace Cyrus
occupation: ['musician', 'backing vocalist', 'bassist', 'vocalist', 'guitarist']
relationship: MEMBER_OF
member_of: Metro Station
</item 9>

<item 10>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 10>
=== END ITEM id=f007be43802e4089 ===

=== ITEM id=3594b6ab7b88d216 ===
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
Megamax C
Megamax C is a K&R implementation of C for the Atari ST, Apple IIgs, and Macintosh computers.

 Sold by Megamax, Inc., based in Richardson, Texas, the package includes a one-pass compiler, linker, editor, resource construction kit, and documentation.

 In the early days of the Atari ST, Megamax C was the primary competitor to the Alcyon C compiler from Digital Research which was included in the official
developer kit from Atari.

 According to a review in "Antic" magazine, Megamax C compiled a small benchmark program six times faster than Digital Research's compiler.
</item 1>

<item 2>
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
</item 2>

<item 3>
Page 6
Page 6 (subtitled "Atari Users Magazine", and later known as New Atari User) was an independent British publication aimed at users of Atari home computers.

 It was published between 1982 and 1998.

 The computer magazine supported both the Atari 8-bit family of computers (400/800/XL/XE) and the Atari ST range.
</item 3>

<item 4>
DTACK Grounded
DTACK Grounded was a computer hobbyist newsletter published from July 1981 to September 1985 by Hal W. Hardenberg.

 Subtitled "The Journal of Simple 68000 Systems", the newsletter was dedicated to the proposition that the Motorola 68000 CPU could be used to build simple fast
computers.

 In 1981 this was a revolutionary idea.

 This was before 68000-based personal computers like the Apple Macintosh, Commodore Amiga, and Atari ST had been created.

 In 1981 Motorola was marketing the 68000 solely as a CPU for high-end Unix workstations.

 Hal believed that the 68000 could be used as a simple embedded microprocessor as well, and used the newsletter to explain how to do that.
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
Happy drives
Happy drives are series of disk drive enhancements for the Atari 8-bit and Atari ST computer families produced by a small company Happy Computers.

 Happy Computers is most noted for the add-in boards for the Atari 810 and Atari 1050 disk drives, which achieved a tremendous speed improvement for reading and
writing, and for the ability to "back up" floppies.

 Happy's products were among the most popular Atari computer add-ons.

 They were still in use and active in the aftermarket as of 2009.
</item 6>

<item 7>
Atari Corporation
Atari Corporation was an American manufacturer of computers and video game consoles from 1984 to 1996.

 Atari Corp. was founded in July 1984 when Warner Communications sold the home computing and game console divisions of Atari, Inc. to Jack Tramiel.

 Its chief products were the Atari ST, , Atari 7800, Atari Lynx, and Atari Jaguar.

 The company reverse merged with JTS Inc. in 1996, becoming a small division, which itself closed when JTS liquidated the IP to Hasbro Interactive in 1998.
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
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 10>
=== END ITEM id=3594b6ab7b88d216 ===

=== ITEM id=5007e50357b637d2 ===
Gold sentences and facts:
Sentence 0:  His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed according to its experimental approach.
  Fact 0.0: His experimental works are well known in the Malayalam Cinema industry.
  Fact 0.1: His experimental works are appropriately discussed in the Malayalam Cinema industry.
  Fact 0.2: His debut movie is Jalachhayam.
  Fact 0.3: Jalachhayam was released in 2010.
  Fact 0.4: Jalachhayam was well discussed according to its experimental approach.
Sentence 1: Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA, Britain and Germany.
  Fact 1.0: Jacques Feyder was born on 21 July 1885.
  Fact 1.1: Jacques Feyder died on 24 May 1948.
  Fact 1.2: Jacques Feyder was a Belgian actor.
  Fact 1.3: Jacques Feyder was a Belgian screenwriter.
  Fact 1.4: Jacques Feyder was a Belgian film director.
  Fact 1.5: Jacques Feyder worked principally in France.
  Fact 1.6: Jacques Feyder also worked in the USA.
  Fact 1.7: Jacques Feyder also worked in Britain.
  Fact 1.8: Jacques Feyder also worked in Germany.

Retrieved items:
<item 1>
Jacques Feyder
Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA,
Britain and Germany.

 He was a leading director of silent films during the 1920s, and in the 1930s he became associated with the style of poetic realism in French cinema.

 He adopted French nationality in 1928.
</item 1>

<item 2>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 2>

<item 3>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 3>

<item 4>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 4>

<item 5>
Sathish Kalathil
Sathish Kalathil (മലയാളം: ) is an Indian film and documentary Director and Producer in malayalam.

 He is also Story Writer, and Lyricist.

 His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed
according to its experimental approach.

 Actually, his first venture was in 2008 named Veena Vaadanam, a documentary film about Art (Painting) and that was his first experimental entry in movie.

 In 2012, he directed Laloorinu Parayanullathu, a social committed documentary film about Municipal solid waste.
</item 5>

<item 6>
Pension Mimosas
Pension Mimosas is a 1935 French film directed by Jacques Feyder.

 Based on an original scenario by Feyder and Charles Spaak, it is a psychological drama set largely in a small hotel on the Côte d'Azur, and it provided
Françoise Rosay with one of the most substantial acting roles of her career.
</item 6>

<item 7>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 7>

<item 8>
Alain Jessua
Alain Jessua (born 16 January 1932) is a French film director and screenwriter.

 He directed ten films between 1956 and 1997.

 He worked as assistant director for Jacques Becker on the set of "Casque d'or", with Max Ophüls for "Madame de..." and "Lola Montès" and with Marcel Carné on
"Wasteland". "

Léon la lune" his first short film won the influential Prix Jean Vigo in 1957.

 He directed first feature film in 1963 "La vie à l'envers" that won Best First Film at Venice Film Festival, in 1964.
</item 8>

<item 9>
L'Atlantide (1921 film)
L'Atlantide is a 1921 French-Belgian silent film directed by Jacques Feyder, and the first of several adaptations of the best-selling novel "L'Atlantide" by
Pierre Benoit.
</item 9>

<item 10>
Thérèse Raquin (1928 film)
Thérèse Raquin is a 1928 drama film directed by Jacques Feyder.

 It is the third silent film adaptation of the novel of the same name by Émile Zola.

 The film stars Gina Manès as Thérèse Raquin, Wolfgang Zilzer as Monsieur Raquin, and Jeanne Marie Laurent as Madame Raquin.

 The décors of the Paris suburbs for the film were built by André Andrejew.

 The film was produced by Deutsche Film Union in Germany, with German and French actors, in a French-German co-production, to be later released at the same time
in France as "Thérèse Raquin" and Germany as Du sollst nicht ehebrechen!

.

 As no words were spoken, both versions differed only with the language of intertitles.

 The British title at the time of the film's original release was Thou Shalt Not.

 This is last silent film imports distributed by Warner Bros.' newly acquired First National subsidiary.

 No dialong with music score and sound effects.
</item 10>
=== END ITEM id=5007e50357b637d2 ===
