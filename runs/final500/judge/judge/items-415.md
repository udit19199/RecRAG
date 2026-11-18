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

=== ITEM id=abba312d0b1c33cd ===
Gold sentences and facts:
Sentence 0: The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the University of California system.
  Fact 0.0: The University of California, Santa Barbara is commonly referred to as UC Santa Barbara or UCSB
  Fact 0.1: The University of California, Santa Barbara is a public research university
  Fact 0.2: The University of California, Santa Barbara is one of the 10 campuses of the University of California system
Sentence 1:  Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest general-education campus in the system.
  Fact 1.0: UCSB traces its roots back to 1891
  Fact 1.1: UCSB's roots as an independent teachers' college date to 1891
  Fact 1.2: UCSB joined the University of California system in 1944
  Fact 1.3: UCSB is the third-oldest general-education campus in the University of California system
Sentence 2:  It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.
  Fact 2.0: UCSB was founded in 1991
  Fact 2.1: UCSB was founded by the renowned philanthropist Hakim Said
  Fact 2.2: Hakim Said is a philanthropist of the Hamdard Foundation

Retrieved items:
<item 1>
Tunku Abdul Rahman University College
Tunku Abdul Rahman University College, abbreviated as TAR UC, is a non-profit, comprehensive, private university college in Malaysia.

 Named after the first Prime Minister of Malaysia, YTM Tunku Abdul Rahman Putra Al-Haj, the institution was founded in 1969 as Tunku Abdul Rahman College or TAR
College then by the Malaysian Chinese Association (MCA).

 In 2013, the college was upgraded to a university college.

 TAR UC is one of the prestigious private university college in Malaysia, renowned for its accounting programme.
</item 1>

<item 2>
Santa Barbara City College
Santa Barbara City College (SBCC) is a two-year community college that opened in 1909 and is located on a 74 acre campus in the city of Santa Barbara,
California, USA.

 SBCC was named co-winner of the 2013 Aspen Institute Prize for Community College Excellence, earning it a #1 national ranking.

 The college was recognized for its quality and focus in four areas: facilitating underrepresented and minority student success, student learning outcomes,
degree completion and transfer rates, and labor market success in securing good jobs after college.

 Santa Barbara City College was established by the Santa Barbara High School District in 1909, making it one of the oldest community colleges in California.

 The college was discontinued shortly after World War I, and its work largely taken over by the Santa Barbara State Normal School, which became the Santa
Barbara State College, and later, the University of California, Santa Barbara.
</item 2>

<item 3>
Mission Santa Barbara
Mission Santa Barbara, also known as Santa Barbara Mission, is a Spanish mission founded by the Franciscan order near present-day Santa Barbara, California.

 It was founded by Padre Fermín Lasuén on December 4, 1786, the feast day of Saint Barbara, as the tenth mission for the religious conversion of the indigenous
local Chumash-Barbareño tribe of Native American people.

 The mission is the namesake of the city of Santa Barbara as well as of Santa Barbara County.
</item 3>

<item 4>
University of California, Santa Barbara Library
The University of California, Santa Barbara Library is the university library system of the University of California, Santa Barbara in Santa Barbara,
California.

 The Library includes four facilities: Two libraries (the Main Library and the Music Library) and two annexes (Annex I and Annex II).

 The library has some three million print volumes, 30,000 electronic journals, 34,450 e-books, 900,055 digitized items, five million cartographic items
(including some 467,000 maps and 3.2 million satellite and aerial images), more than 3.7 million pieces of microform, 167,500 sound recordings, and 4,100
manuscripts.

 The Library states that it holds 3.2 mi of manuscript and archival collections.

 According to data collected by the American Library Association, as of July 2010 the UC Santa Barbara Library holds the 99th largest library collection in
America, reporting 2,948,999 volumes.
</item 4>

<item 5>
Hamdard University
Hamdard University (Urdu: ) is a private research university with campuses in Karachi and Islamabad, Pakistan.

 It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.

 Hamdard is one of the first and the oldest private institutions of higher education in Pakistan.

 In Karachi, Hamdard University is the largest private research university with a campus area of over 350 acres.
</item 5>

<item 6>
University of Arizona
The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.

 Founded in 1885, the UA was the first university in the Arizona Territory.

 The university operates two medical schools (University of Arizona College of Medicine - Tucson and the University of Arizona College of Medicine - Phoenix)
and is affiliated with the region's only academic medical centers (Banner - University Medical Center Tucson and Banner - University Medical Center Phoenix).

 The university is also home to the James E. Rogers College of Law and numerous other nationally ranked graduate and professional schools.

 During the 2016–2017 academic year, there was a total enrollment of 43,625 students, including 34,072 undergraduates The University of Arizona is governed by
the Arizona Board of Regents.

 The University of Arizona is one of the elected members of the Association of American Universities (an organization of North America's premier research
institutions) and is the only representative from the state of Arizona to this group.
</item 6>

<item 7>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 7>

<item 8>
Santa Barbara High School
Santa Barbara Senior High School, "Home of the Dons," is situated on a 40 acre campus in Santa Barbara, California and is part of the Santa Barbara Unified
School District.

 One of the oldest high schools in California, it was established in 1875, and moved to its current site in 1924.

 Until the creation of two rival high schools in 1959 and 1966, it was the sole public high school serving the city of Santa Barbara.

 In November 2005, its 18th-century Spanish influenced design was named an official California and City historic landmark.

 Today, Santa Barbara High School has an enrollment of roughly 2,200 pupils and certified staff numbering 108.
</item 8>

<item 9>
University of California, Santa Barbara
The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the
University of California system.

 The main campus is located on a 1022 acre site near Isla Vista, California, United States, 8 mi from Santa Barbara and 100 mi northwest of Los Angeles.

 Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest
general-education campus in the system.
</item 9>

<item 10>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 10>
=== END ITEM id=abba312d0b1c33cd ===

=== ITEM id=244b03d3e44f95c0 ===
Gold sentences and facts:
Sentence 0: John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).
  Fact 0.0: John H. Garabedian is an American radio personality.
  Fact 0.1: John H. Garabedian is an American disc jockey.
  Fact 0.2: John H. Garabedian was born on December 20, 1941.
  Fact 0.3: John H. Garabedian is age 75.
Sentence 1: Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
  Fact 1.0: Charles A. Agemian was born July 22, 1909.
  Fact 1.1: Charles A. Agemian's life ended March 30, 1996.
  Fact 1.2: Charles A. Agemian was an American banker.
  Fact 1.3: Charles A. Agemian took early retirement from Chase Manhattan Bank.
  Fact 1.4: Charles A. Agemian was executive vice president of operations at Chase Manhattan Bank.
  Fact 1.5: Charles A. Agemian became chairman of the Hackensack Trust Co.
  Fact 1.6: Charles A. Agemian became chief executive officer of the Hackensack Trust Co.
  Fact 1.7: The Hackensack Trust Co. was later renamed Garden State National Bank.

Retrieved items:
<item 1>
Barry Ackerley headed Ackerley Group and owned the Seattle SuperSonics (1983–2001); he died after a stroke in 2011. Al Carter was an award-winning
Texas/Oklahoma sportswriter. André Laguerre led Sports Illustrated (1960–74) and introduced its Swimsuit Issue. Dick “Scoop” Gordon was a longtime sports
journalist. Cork fiddler John Murphy learned from his father. Galveston gang leader Johnny Jack N
</item 1>

<item 2>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 2>

<item 3>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 3>

<item 4>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 4>

<item 5>
Big Wilson
Big Wilson (born Malcolm John Wilson, Jr.; October 3, 1924 – October 5, 1989) was an American radio personality.

 He worked as a disc jockey at WNBC AM in New York City from the early 1960s until 1974 and moved to Miami in 1975 where he worked for WIOD and WCIX-TV.

 He was one of the last two hosts of the NBC network program Monitor.

 Prior to moving to NYC, Wilson and his wife Jody lived in a riverfront home in Rocky River, Ohio.

 He commuted to Cleveland where he was the leading radio personality for some years.

 Tim Conway and other well-known persons visited their home.
</item 5>

<item 6>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 6>

<item 7>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 7>

<item 8>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 8>

<item 9>
Don Sherwood (DJ)
Don Sherwood (September 7, 1925 – November 6, 1983) was an American radio personality.

 He was a San Francisco, California, disc jockey during the 1950s and 1960s.

 Billed as "The World's Greatest Disc Jockey," Sherwood spent most of his career hosting a 6-9 a.m. weekday program on KSFO in San Francisco (560 kHz, 5000
watts), which was then owned by the singing cowboy actor Gene Autry.
</item 9>

<item 10>
Radio personality
A radio personality (American English) or radio presenter (British English), commonly referred to as a "disc jockey" or "DJ" for short, is a person who has an
on-air position in radio broadcasting.

 A radio personality that hosts a radio show is also known as a radio host, and in India and Pakistan as a radio jockey.

 Radio personalities who introduce and play individual selections of recorded music are known as disc jockeys.

 The term has evolved to also describe a person who mixes a continuous flow of recorded music in real time.

 Broadcast radio personalities may include talk radio hosts, AM/FM radio show hosts, and satellite radio program hosts.

 Notable radio personalities include pop music radio hosts Martin Block, Alan Freed, Dick Clark, Delilah Luke, Ameen Sayani, Wolfman Jack, and Casey Kasem,
shock jocks such as Don Imus and Howard Stern, as well as sports talk hosts such as Mike Francesa and political talk hosts such as Rush Limbaugh.
</item 10>
=== END ITEM id=244b03d3e44f95c0 ===

=== ITEM id=60b3017a38213875 ===
Gold sentences and facts:
Sentence 0:  He is the main vocalist of the boy group Highlight.
  Fact 0.0: The person described is the main vocalist of the boy group Highlight.
Sentence 1:  She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls' Generation, in 2007.
  Fact 1.0: The person described was a trainee at S.M. Entertainment's Starlight Academy during her middle school years.
  Fact 1.1: The person described debuted as a member of Girls' Generation, the girl group of S.M. Entertainment, in 2007.
  Fact 1.2: The person described debuted in Girls' Generation after her trainee years.

Retrieved items:
<item 1>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 1>

<item 2>
Yang Yo-seob discography
The discography of South Korean singer-songwriter, musical actor Yang Yo-seob consists of one extended play and two singles.
</item 2>

<item 3>
Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo"), also known as simply SJ or SUJU, is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 3>

<item 4>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 4>

<item 5>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 5>

<item 6>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 6>

<item 7>
Graph query row:
person: Kim Tae-yeon
music_group: Girls' Generation
</item 7>

<item 8>
Yang Yo-seob
Yang Yo-seob (born January 5, 1990), more commonly known as Yoseob, is a South Korean singer and musical actor.

 He is the main vocalist of the boy group Highlight.
</item 8>

<item 9>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 9>

<item 10>
Graph query row:
person: Yang Yo-seob
music_group: Highlight
</item 10>
=== END ITEM id=60b3017a38213875 ===

=== ITEM id=1a9d4f08bea29514 ===
Gold sentences and facts:
Sentence 0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
  Fact 0.0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia.
  Fact 0.1: The 1997 European Athletics Junior Championships were held on July 24-27.
Sentence 1: Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.
  Fact 1.0: Ljubljana is the capital of Slovenia.
  Fact 1.1: Ljubljana is the largest city of Slovenia.
  Fact 1.2: Ljubljana is also known by other, historical names.

Retrieved items:
<item 1>
European Athletics U20 Championships
The European Athletics U20 Championships (formerly named the European Athletics Junior Championships up to 2015) are the European championships for athletes who
are 19 years of age or under, which is the age range recognised by the IAAF as junior athletes.

 The event is currently organized by the European Athletic Association.
</item 1>

<item 2>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 2>

<item 3>
2015 European Athletics Junior Championships
The 2015 European Athletics Junior Championships were the 23rd edition of the biennial European U20 athletics championships.

 They were held in Eskilstuna, Sweden from 16 July to 19 July.
</item 3>

<item 4>
1997 European Athletics Junior Championships
The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
</item 4>

<item 5>
Ljubljana
Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.

 It has been the cultural, educational, economic, political, and administrative center of independent Slovenia since 1991.

 Its central geographic location within Slovenia, transport connections, concentration of industry, scientific and research institutions, and cultural tradition
are contributing factors to its leading position.
</item 5>

<item 6>
1968 European Junior Games
The 1968 European Junior Games was the third edition of the biennial athletics competition for European athletes aged under twenty.

 It was the final edition of the competition under that name and was succeeded by the European Athletics Junior Championships in 1970.

 The event was held in Leipzig, part of the German Democratic Republic at that time.

 The competition was mainly between Eastern European countries, with Belgium, Austria, Netherlands and Greece being present from the west.
</item 6>

<item 7>
Amela Terzić
Amela Terzić (Serbian Cyrillic: Амела Терзић, born 2 January 1993 in Priboj, FR Yugoslavia) is a Serbian middle-distance runner.

 She won two gold medals at 2011 European Athletics Junior Championships in Tallinn and was the junior champion at the 2012 European Cross Country
Championships.

 She has also been a medallist in the 1500 metres at the World Junior and World Youth Championships and a gold medalist at the 2013 European U23 Championship.
</item 7>

<item 8>
1996 IPB Czech Indoor – Doubles
Jonas Björkman and Javier Frana was the defending champions but did not compete that year.
</item 8>

<item 9>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 9>

<item 10>
1964 European Junior Games
The 1964 European Junior Games was the first edition of the biennial athletics competition for European athletes aged under twenty.

 It was an unofficial competition without sanction from the European Athletic Association.

 The event was held at the 10th-Anniversary Stadium in Warsaw, Poland.

 The competition eventually led to the creation of the European Athletics Junior Championships in 1970.
</item 10>
=== END ITEM id=1a9d4f08bea29514 ===

=== ITEM id=3c3be440169ada0f ===
Gold sentences and facts:
Sentence 0: Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
  Fact 0.0: Alan Crosland was born on August 10, 1894.
  Fact 0.1: Alan Crosland died on July 16, 1936.
  Fact 0.2: Alan Crosland was an American stage actor.
  Fact 0.3: Alan Crosland was an American film director.
Sentence 1: Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.
  Fact 1.0: Kurt Voss was born Kurt Christopher Peter Wössner.
  Fact 1.1: Kurt Voss is an American film director.
  Fact 1.2: Kurt Voss is an American screenwriter.
  Fact 1.3: Kurt Voss is an American musician-songwriter.

Retrieved items:
<item 1>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 1>

<item 2>
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
</item 2>

<item 3>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 3>

<item 4>
Arthur Tedder, 1st Baron Tedder
After the war he served as Chief of the Air Staff, in which role he advocated increased recruiting in the face of many airmen leaving the service, doubled the
size of RAF Fighter Command and implemented arrangements for the Berlin Airlift in 1948.

 After the war he held senior positions in business and academia.
</item 4>

<item 5>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 5>

<item 6>
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
</item 6>

<item 7>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 7>

<item 8>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 8>

<item 9>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 9>

<item 10>
Al Ulmer
Alfred C. Ulmer, Jr. was born in Jacksonville, Florida in August 1916.

 He was of Swiss extraction on his father's side, his father having been born in Zurich.

 Ulmer graduated from Princeton University in 1939 and joined the United States Navy prior to the start of World War II, ultimately becoming a major head of
intelligence operations during World War II.

 He married Doris Gibson Bridges and had three sons (Alfred III, James and Nicolas) and a daughter (Marguerite).

 He received the Intelligence Medal of Merit when he retired from his position in 1962.

 Ulmer then went on to business and in the 1980s joined the Swiss banking firm Lombard Odier et Cie. in Geneva, later setting up Lombard Odier's operations in
Bermuda.

 He died on June 22, 2000 in Virginia Beach, Virginia.
</item 10>
=== END ITEM id=3c3be440169ada0f ===

=== ITEM id=cf01002df79a77db ===
Gold sentences and facts:
Sentence 0:  She was born in Guangzhou, China.
  Fact 0.0: The woman referred to as "she" was born in Guangzhou, China.
Sentence 1:  These videos are an overview and recap of each year's viral videos, events, memes, and music.
  Fact 1.0: The videos being described are an overview of each year's viral videos, events, memes, and music.
  Fact 1.1: The videos being described are a recap of each year's viral videos, events, memes, and music.

Retrieved items:
<item 1>
Graph query row:
person: Wendy Ayche
relationship: BORN_IN
related: Guangzhou
</item 1>

<item 2>
YouTube Rewind
YouTube Rewind is a video series produced and created by YouTube and Portal A Interactive.

 These videos are an overview and recap of each year's viral videos, events, memes, and music.

 Each year, the number of YouTube celebrities featured in the video, as well as the presentation of the series, have increased.

 The latest episode of "YouTube Rewind" was released on December 8, 2016.
</item 2>

<item 3>
Wengie
Wendy Ayche (born January 9 1986) known professionally as Wengie, is an Australian YouTube personality and vlogger.

 She was born in Guangzhou, China.

 She got the nickname "Wengie" at a dance class when two people gave her a nickname inspired by her Chinese name, Wén Jié.

 Since starting her channel on February 11 2013, her videos have received over 281 million views, and her channel has accumulated over 11 followers In the
January 2016 edition of "ElleGirl Japan", she was nominated as a channel to watch.

 In August 2016, she was ranked the 5th Fastest growing channel in the world.

 Ayche was featured in the annual YouTube Rewind in 2016 and was also singled out by Google as one of the top beauty creators in the Asia region.

 Ayche reached 5 million YouTube subscribers in January 2017, and her channel is currently in the 12th most subscribed How-To & Style Channel on YouTube.

 She is also currently a board member of the Internet Creators Guild, a non-profit focused on providing the protection, representation and guidance to online
creators.
</item 3>

<item 4>
Lilly Singh
Lilly Singh (born 26 September 1988) is a Canadian YouTube personality, vlogger, comedian, writer, and actress.

 She is better known by her YouTube username IISuperwomanII.

 Since beginning her channel in October 2010, her videos have received over 2 billion views, and her channel has accumulated over 12 million subscribers.

 In 2016, she was ranked 3rd on the "Forbes" list of the world's highest paid YouTube stars (behind Roman Atwood and PewDiePie), earning a reported $7.5 million
in 2016.

 Singh has featured in the annual YouTube Rewind every year since 2014.

 She ranked 1st on 2017 Forbes Top Influencers List in the entertainment category.
</item 4>

<item 5>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 5>

<item 6>
Zhu (musician)
Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of
2014, signed to Mind of a Genius Records and Columbia Records.

 Until mid-2014, Zhu remained anonymous, asking to be judged by his music alone.

 His debut album, "Generationwhy", was released on 29 July 2016.
</item 6>

<item 7>
Wayne Goss (make-up artist)
Wayne Alan Goss (born 4 March 1978) is an English makeup artist, YouTube personality and entrepreneur.

 Best known for his "Wayne Goss" channel on YouTube, he is the creator and owner of the self-eponymous makeup brush line launched in September 2013.

 As of November 2016, Goss' main YouTube channel had over 2.7 million subscribers and had had 344 million views.
</item 7>

<item 8>
Kevin Nalty
Kevin 'Nalts' Nalty (born May 12, 1969) is a YouTube comedian and partner based in Doylestown, Pennsylvania better known under his YouTube username Nalts.

 Nalts began on YouTube as one of the top-20 most-viewed comedy channels, and collaborates with many of today's top YouTube personalities.

 He has more than 1,000 videos which, as of April 18, 2016, have been viewed more than 297 million times on YouTube alone.

 He has been ranked as one of YouTube's Most Subscribed users.

 He is the author of "Beyond Viral: How to Attract Customers, Promote Your Brand, and Make Money with Online Video (Wiley & Sons, 2010).
</item 8>

<item 9>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 9>

<item 10>
CVX Live
CVX Live (Creator Viewer Experience Live) is an event designed to bring online media creators and their fans together.

 It was held for the first time in Orem, Utah, on August 7 to 8, 2015 Originally created by the team behind the "Bored Shorts TV" YouTube channel, the
convention is Utah's first YouTube event, gathering thousands of online YouTube creators and fans together in one place.

 One of the creators, Brett Robers, described it as “...a very unique opportunity to bring YouTube to life.”
</item 10>
=== END ITEM id=cf01002df79a77db ===

=== ITEM id=405383f475b69a2c ===
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
Pennsylvania Route 145
MacArthur Road is the location of the main commercial center of the Lehigh Valley.

 North of Eagle Point, PA 145 becomes a two-lane undivided road that parallels the Lehigh River, crossing the river into Northampton County at Treichlers.

 The route continues along the east bank of the river and passes through Walnutport before reaching its northern terminus.

 PA 145 is dedicated as the Battle of the Bulge Veterans Memorial Highway in honor of the veterans who fought in the Battle of the Bulge.
</item 1>

<item 2>
Pennsylvania Canal (Delaware Division)
as part of the solution to the United States' first energy crisis.

 The Commonwealth of Pennsylvania built the Delaware canal to feed anthracite stone coal to energy-hungry Philadelphia as part of its transportation
infrastructure building plan known as the Main Line of Public Works—a legislative initiative creating a collection of self-reinforcing internal improvements to
commercial transportation capabilities.
</item 2>

<item 3>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 3>

<item 4>
Jersey barrier
A Jersey barrier or Jersey wall is a modular concrete or plastic barrier employed to separate lanes of traffic.

 It is designed to minimize vehicle damage in cases of incidental contact while still preventing the crossover case of a head-on collision.

 Jersey barriers are also used to reroute traffic and protect pedestrians and workers during highway construction, as well as temporary and semi-permanent
protections against landborne attack such as suicide vehicle bombs.

 A Jersey barrier is also known in the western United States as K-rail, or K-wall, a term borrowed from the California Department of Transportation
specification for temporary concrete traffic barriers, or colloquially as a Jersey bump.

 Plastic water-filled barriers of the same general shape are also now commonly called Jersey barriers.
</item 4>

<item 5>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 5>

<item 6>
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
</item 6>

<item 7>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 7>

<item 8>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 8>

<item 9>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 9>

<item 10>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 10>
=== END ITEM id=405383f475b69a2c ===

=== ITEM id=c16e9daff1477ed0 ===
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
Volvo 164
The Volvo 164 is a 4-door, 6-cylinder luxury sedan unveiled by Volvo at the Paris Motor Show early in October 1968 and first sold as a 1969 model.

 46,008 164s were built before the car was superseded by the 264 in 1975.

 The 164 was Volvo's first venture into the luxury segment since the end of PV 60 production in 1950, and was the first six-cylinder Volvo since the PV800 last
produced in 1958.
</item 1>

<item 2>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 2>

<item 3>
Volvo SI6 engine
Volvo designed the SI6 ("short inline 6") straight-6 automobile engine for use in 2007 models.

 An evolution of the company's long-used straight-5 Volvo Modular engine, which itself is an evolution of the Volvo B6304 straight six engine, the SI6 can be
mounted transversely for front wheel drive or all wheel drive applications.

 Despite the added cylinder and displacement, the engine remains compact, and is in fact 1 mm shorter than the previous straight-5.

 The engine was initially offered in two displacements — a 3.0 L turbocharged version and a 3.2 L naturally aspirated version.

 Both offered variable cam timing, though only the turbo version varies both the intake and exhaust valves.

 On top of the variable cam timing used on the intake cam of the naturally aspirated engine it also had variable valve lift by using Cam Profile Switching
(CPS).
</item 3>

<item 4>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 4>

<item 5>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 5>

<item 6>
Volvo 440/460
The Volvo 440 and 460 are versions of a small family car produced by the Swedish manufacturer Volvo between 1987 and 1996.

 The 440 was introduced in 1987, whilst the 460 followed in 1989.

 They were built at the NedCar factory in Born, The Netherlands.

 The 440 was a five door front wheel drive hatchback and the 460 a saloon.

 They shared many components with the already successful Volvo 480, including floorpan, front and rear suspension, engines, transmission and braking systems.
</item 6>

<item 7>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 7>

<item 8>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 8>

<item 9>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 9>

<item 10>
Volvo P2 platform
The Volvo P2 platform is a global full-size unibody automobile platform developed and produced by Volvo.

 It is designed for single wheelbases and is adaptable to front- or all wheel drive configurations.

 It was developed by the automaker before its 1999 acquisition by Ford Motor Company, and debuted with the 1998 Volvo S80.
</item 10>
=== END ITEM id=c16e9daff1477ed0 ===

=== ITEM id=cea0f91d025e8df7 ===
Gold sentences and facts:
Sentence 0: August or Auguste Brancart (21 July 1851 - 1894?)
  Fact 0.0: August Brancart is also known as Auguste Brancart.
  Fact 0.1: August Brancart was born on 21 July 1851.
  Fact 0.2: The end date given for August Brancart is 1894, with a question mark.
Sentence 1:  He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
  Fact 1.0: Brancart published work of the Decadent movement.
  Fact 1.1: Brancart published "Monsieur Vénus" by Rachilde.
  Fact 1.2: "Monsieur Vénus" by Rachilde is an example of Decadent movement work.
Sentence 2: The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and artificiality.
  Fact 2.0: The Decadent Movement was an artistic and literary movement.
  Fact 2.1: The Decadent Movement was a late 19th-century movement.
  Fact 2.2: The Decadent Movement was centered in Western Europe.
  Fact 2.3: The Decadent Movement followed an aesthetic ideology of excess.
  Fact 2.4: The Decadent Movement followed an aesthetic ideology of artificiality.

Retrieved items:
<item 1>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 1>

<item 2>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 2>

<item 3>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 3>

<item 4>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 4>

<item 5>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 5>

<item 6>
Christian de la Mazière
Christian de la Mazière (August 22, 1922 in Tunis – February 15, 2006) was a journalist and member of the Charlemagne Division of the Waffen SS.

 He is known for discussing his role in the documentary "The Sorrow and the Pity" and also wrote a book titled "The Captive Dreamer".

 At the start of the war, he served in the French Army and was later retained in the military of Vichy France until 1942.

 After being discharged, he worked for the fascist newspaper "Le Pays Libre", joining the Charlemagne Division just before the Liberation of Paris in 1944.

 He was taken prisoner in Pomerania by Polish forces in the Red Army.
</item 6>

<item 7>
Decadent movement
The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and
artificiality.

 The visual artist Félicien Rops's body of work and Joris-Karl Huysmans's novel "Against Nature" (1884) are considered the prime examples of the decadent
movement.

 It first flourished in France and then spread throughout Europe and to the United States.

 The movement was characterized by self-disgust, sickness at the world, general skepticism, delight in perversion and employment of crude humor and a belief in
the superiority of human creativity over logic and the natural world.
</item 7>

<item 8>
Reminiscence of Marie A.
"Reminiscence of Marie A." or "Memory of Marie A." (German: ""Erinnerung an die Marie A."") is a 1920 poem by German poet and playwright Bertold Brecht
(1889-1956) that was first published in his collection "Die Hauspostille" (1927).

 Brecht wrote the poem in his notebook on 21 February 1920 on a train to Berlin.

 The poem is a reminiscence of time spent with a former lover and a kiss beneath a plum tree remembered only because of the memory of a passing white cloud.
</item 8>

<item 9>
Hermann Brunn
Karl Hermann Brunn (1 August 1862 – 20 September 1939) was a German mathematician, known for his work in convex geometry (see Brunn–Minkowski inequality) and in
knot theory.

 Brunnian links are named after him, as his 1892 article "Über Verkettung" included examples of such links.
</item 9>

<item 10>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 10>
=== END ITEM id=cea0f91d025e8df7 ===

=== ITEM id=0797718c654bfe34 ===
Gold sentences and facts:
Sentence 0:  "Pinus" is the sole genus in the subfamily Pinoideae.
  Fact 0.0: Pinus is the sole genus in the subfamily Pinoideae.
Sentence 1:  The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35 unresolved species and many more synonyms.
  Fact 1.0: The Plant List was compiled by the Royal Botanic Gardens, Kew and the Missouri Botanical Garden.
  Fact 1.1: The Plant List accepts 126 species names of pines as current.
  Fact 1.2: The Plant List includes 35 unresolved species of pines.
  Fact 1.3: The Plant List includes many more synonyms.
Sentence 2:  It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
  Fact 2.0: The genus is sometimes considered to have only two species, B. monosperma and B. superba.
  Fact 2.1: The genus is sometimes expanded to include four or five species.

Retrieved items:
<item 1>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 1>

<item 2>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 2>

<item 3>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 3>

<item 4>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Pine
A pine is any conifer in the genus Pinus, , of the family Pinaceae.

 "Pinus" is the sole genus in the subfamily Pinoideae.

 The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35
unresolved species and many more synonyms.
</item 6>

<item 7>
Oak
An oak is a tree or shrub in the genus Quercus ( ; Latin "oak tree") of the beech family, Fagaceae.

 There are approximately 600 extant species of oaks.

 The common name "oak" also appears in the names of species in related genera, notably "Lithocarpus" (stone oaks), as well as in those of unrelated species such
as "Grevillea robusta" (silky oaks) and the Casuarinaceae (she-oaks).

 The genus "Quercus" is native to the Northern Hemisphere, and includes deciduous and evergreen species extending from cool temperate to tropical latitudes in
the Americas, Asia, Europe, and North Africa.

 North America contains the largest number of oak species, with approximately 90 occurring in the United States, while Mexico has 160 species of which 109 are
endemic.

 The second greatest center of oak diversity is China, which contains approximately 100 species.
</item 7>

<item 8>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 8>

<item 9>
Yellow pine
In ecology and forestry, yellow pine refers to a number of pine species which tend to grow in similar plant communities and yield similar strong wood.

 In the Western United States, yellow pine refers to Jeffrey pine or ponderosa pine.

 In the Southern United States, yellow pine refers to longleaf pine, shortleaf pine, slash pine, or Loblolly pine.

 In the United Kingdom, yellow pine refers to Eastern white pine or Scots pine.
</item 9>

<item 10>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 10>
=== END ITEM id=0797718c654bfe34 ===

=== ITEM id=af41f58a4c61d143 ===
Gold sentences and facts:
Sentence 0: Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.
  Fact 0.0: Xingcheng is a county-level city of southwest Liaoning province, China.
  Fact 0.1: Xingcheng's former name was Ningyuan (宁远).
  Fact 0.2: Xingcheng has a population of approximately 140,000 urban inhabitants.
  Fact 0.3: Xingcheng is located on Liaodong Bay, the northern coast of the Bohai Sea.
Sentence 1:  It covers an area of 2046 square kilometers and has a population of about one million.
  Fact 1.0: Xingcheng covers an area of 2046 square kilometers.
  Fact 1.1: Xingcheng has a population of about one million.

Retrieved items:
<item 1>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 1>

<item 2>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 2>

<item 3>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 3>

<item 4>
Huan River
The Huan River (), or Anyang River (安阳河), is a river in Henan, China, and part of the Hai River basin.

 The river rises north of Linzhou in northwestern Henan, and joins the Wei River near Neihuang in the northeast of the province.
</item 4>

<item 5>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
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
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 7>

<item 8>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 8>

<item 9>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 9>

<item 10>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 10>
=== END ITEM id=af41f58a4c61d143 ===

=== ITEM id=4765ba2ebef18af0 ===
Gold sentences and facts:
Sentence 0: George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
  Fact 0.0: George Albert Smith Sr. was born on April 4, 1870.
  Fact 0.1: George Albert Smith Sr. died on April 4, 1951.
  Fact 0.2: George Albert Smith Sr. was an American religious leader.
  Fact 0.3: George Albert Smith Sr. served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
Sentence 1: Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.
  Fact 1.0: Wong Jing was born on 3 May 1955.
  Fact 1.1: Wong Jing is a Hong Kong film director.
  Fact 1.2: Wong Jing is a Hong Kong film producer.
  Fact 1.3: Wong Jing is a Hong Kong actor.
  Fact 1.4: Wong Jing is a Hong Kong presenter.
  Fact 1.5: Wong Jing is a Hong Kong screenwriter.

Retrieved items:
<item 1>
Angelino Apelar
Angelino "Andy" Baldemor Apelar (June 16, 1927 - January 23, 2006) was an Evangelical Christian leader in the Filipino American community, and a founder of the
Association of Filipino Churches (AFC) of the Christian and Missionary Alliance.

 He had served as a missionary/pastor and radio broadcaster in the Philippines before moving to the United States to start the AFC.

 He served as president for several terms.

 He graduated from the Far Eastern Bible Institute and Seminary, where he also met his future wife, Purita Palomar, a member of one of Capiz province's
influential families.

 They have two children: Faith and Jemuel Apelar.
</item 1>

<item 2>
Mario Gaoa
Mario Gaoa (born 1971) is a New Zealand actor, writer and director, best known as a member of the Naked Samoans comedy group.

 He is of Samoan descent.

 As part of the group he has appeared in the film Sione's Wedding; provided the voices of Sione Tapili and God in the animated series Bro'Town, which he also
co-writes; and acted in various Naked Samoans comedic theatre performances.

 He has also appeared in the film Nightmare Man and briefly in the television series .
</item 2>

<item 3>
The Conman
The Conman is a 1998 Hong Kong action comedy film directed by Wong Jing and starring Andy Lau, Athena Chu and Nick Cheung.

 Despite the Chinese title, which translates as "Knight of Gamblers 1999", Andy Lau does not reprise his role as the "Knight of Gamblers" from the "God of
Gamblers" series, which was also directed by Wong Jing.

 The film was followed by a sequel "The Conmen in Vegas", which Lau and Cheung return with new cast members Natalis Chan, Kelly Lin, Meggie Yu and Alex Man.
</item 3>

<item 4>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 4>

<item 5>
Feng Xingxi
Feng Xingxi (馮行襲) (died July 31, 910), courtesy name Zhengchen (正臣), formally Prince Zhongjing of Changle (長樂忠敬王), was a warlord late in the Chinese dynasty
Tang Dynasty who later became a subject of the succeeding Later Liang.

 He was tall and strong and known as "Green Face Feng" for his green birthmark on his face.
</item 5>

<item 6>
Ba-Jia-Jiang
Ba Jia Jiang (八家將) originated from the Chinese folk beliefs and myths, and usually generally refers to eight members of the godly realm.

 The general understanding of the origin of Ba Jia Jiang is that it is derived from the existence of eight generals who performed exorcism of evil spirits for
the Wufu Emperor (五福大帝).

 These eight generals became revered as the gods of the underworld, and are represented as the bodyguards or attendants for the temples of the nether Gods such
as the Dongyue Emperor (東獄大帝), Yama (King of Hell, 閻羅王) and Cheng Huang (the City Gods, 城隍).

 Gradually Ba Jia Jiang evolved to appear also as bodyguards to Wang Ye (Royal Lord, 王爺) and Matsu (媽祖), and at many other temples.

 Later on, believers at those temples dressed up as Ba Jia Jiang in order to defend the Gods.
</item 6>

<item 7>
George A. Smith
George Albert Smith (June 26, 1817 – September 1, 1875) (known throughout his life as George A. Smith) was an early leader in the Latter-day Saint movement.

 He served in the Quorum of the Twelve Apostles and as a member of the First Presidency of The Church of Jesus Christ of Latter-day Saints (LDS Church).
</item 7>

<item 8>
Joey Bishop
Joseph Abraham Gottlieb (February 3, 1918 – October 17, 2007), known professionally as Joey Bishop, was an American entertainer who appeared on television as
early as 1948 and eventually starred in his own weekly comedy series playing a talk show host, then later hosted a late night talk show.

 He later became a member of the "Rat Pack" with Frank Sinatra, Peter Lawford, Sammy Davis, Jr., and Dean Martin.
</item 8>

<item 9>
Chao-Li Chi
Chao-Li Chi (; April 5, 1927 – October 16, 2010) was a Shanxi-born actor and dancer who worked extensively in American television, including his best known role
as Chao-Li, the faithful majordomo and chauffeur of Jane Wyman's character in "Falcon Crest".

 Additionally, his film credits include "Big Trouble in Little China", "The Joy Luck Club", "The Nutty Professor", "Wedding Crashers" and "The Prestige".

 He was featured in the short film by Maya Deren, "Meditation on Violence", in 1948.
</item 9>

<item 10>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 10>
=== END ITEM id=4765ba2ebef18af0 ===
