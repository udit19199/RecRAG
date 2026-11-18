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

=== ITEM id=fced22ad913cfa0f ===
Gold sentences and facts:
Sentence 0: Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
  Fact 0.0: Janet Williams is an American soprano.
  Fact 0.1: Janet Williams has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Theatre des Champs-Elysees, Opera de Lyon, Nice Opera, Theatre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre.
  Fact 0.2: Janet Williams has performed in concerts throughout Europe, North America, Canada, Israel and Japan.
  Fact 0.3: Janet Williams has performed with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, Rene Jacobs, Marek Janowski, Neeme Jarvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
Sentence 1: Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized solo pianist, chamber music performer, and conductor.
  Fact 1.0: Vladimir Davidovich Ashkenazy's Russian name is Vladimir Davidovich Ashkenazi.
  Fact 1.1: Vladimir Davidovich Ashkenazy was born on 6 July 1937.
  Fact 1.2: Vladimir Davidovich Ashkenazy is an internationally recognized solo pianist.
  Fact 1.3: Vladimir Davidovich Ashkenazy is an internationally recognized chamber music performer.
  Fact 1.4: Vladimir Davidovich Ashkenazy is an internationally recognized conductor.
Sentence 2:  His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
  Fact 2.0: Vladimir Davidovich Ashkenazy's recordings have earned him five Grammy awards.
  Fact 2.1: Vladimir Davidovich Ashkenazy has received Iceland's Order of the Falcon.

Retrieved items:
<item 1>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 1>

<item 2>
The Impeccable Mr. Wilson
The Impeccable Mr. Wilson is an album by American jazz pianist Teddy Wilson featuring performances recorded in 1956 for the Norgran label.
</item 2>

<item 3>
Don Williams discography
This is a detailed discography for American country music artist Don Williams.
</item 3>

<item 4>
Fred Hersch
Fred Hersch (born October 21, 1955) is an American jazz pianist and educator.

 He has performed solo and led his own groups, including the Pocket Orchestra consisting of piano, trumpet, voice, and percussion.

 He was the first person to play weeklong engagements as a solo pianist at the Village Vanguard in New York City.

 He has recorded more than 70 of his jazz compositions.

 Hersch has been nominated for several Grammy Awards, and, as of December 2014, had been on the Jazz Studies faculty of the New England Conservatory since 1980
(with breaks).
</item 4>

<item 5>
Grammy Award for Best Male Rap Solo Performance
The Grammy Award for Best Male Rap Solo Performance was an honor presented to male recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958, and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 5>

<item 6>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 6>

<item 7>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 7>

<item 8>
Grammy Award for Best Female Rap Solo Performance
The Grammy Award for Best Female Rap Solo Performance was an honor presented to female recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958 and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 8>

<item 9>
Janet Williams (soprano)
Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera,
Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco
Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with
conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard,
Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
</item 9>

<item 10>
Robert Glasper
Robert Glasper (born April 6, 1978, in Houston, Texas) is an American pianist and record producer.

 He has been nominated for 6 Grammys, has won 3 Grammy Awards and is currently nominated for an Emmy Award.

 His 2012 album "Black Radio" won the Grammy Award for Best R&B Album at the 55th Grammy Awards.

 His 2014 album "Black Radio 2" won the Grammy Award for Best Traditional R&B Performance at the 56th Grammy Awards.

 The song "These Walls" from Kendrick Lamar's album "To Pimp A Butterfly" won Best Rap/Sung Collaboration at the 57th Grammy Awards, on which Glasper plays
keys.

 The soundtrack for the film "Miles Ahead" won Best Soundtrack Compilation at the 58th Grammy Awards, for which Glasper was a producer.

 The song "Letter To The Free", written with Common, is nominated for an Emmy Award for Best Original Song in the Ava Duvernay documentary film "13th" (Netflix)
at the 2017 Emmys.
</item 10>
=== END ITEM id=fced22ad913cfa0f ===

=== ITEM id=b1c306df0afc5dda ===
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
2013–14 Real Sociedad season
The 2013–14 season was Real Sociedad's 67th season in La Liga.

 Real Sociedad finished 7th in the league and reached the semifinals of the Copa del Rey.

 The Basque failed to make it out of the group stages of the UEFA Champions League.
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
Lippo Hertzka
Lippo Hertzka (19 November 1904 – 14 March 1951) was a Hungarian football player and manager.

 He played for Essener Turnerbund, MTK Budapest and Real Sociedad.

 After retiring, he coached seven teams, including Real Sociedad and Real Madrid, a team which he coached for 2 years (1930–1932) and led to an undefeated La
Liga championship during the 1931-32 season, which meant the first La Liga title for the "white" squad.

 He also won two league titles in Portugal for Benfica.
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
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 6>

<item 7>
C.D. Real Sociedad
Club Deportivo Real Sociedad, commonly known as Real Sociedad (] ), is a Honduran football club based on Tocoa, Colón, Honduras.
</item 7>

<item 8>
Real Sociedad
Real Sociedad de Fútbol, S.A.D., more commonly referred to as Real Sociedad (] ; "Royal Society") or La Real, is a Spanish football club based in the city of
San Sebastián, Basque Country, founded on 7 September 1909.

 It plays its home matches at the 32,000-capacity Anoeta Stadium.

 Real Sociedad won the Liga title in 1980–81 and 1981–82, and last finished runners-up in 2002–03.

 The club has also won the Copa del Rey twice, in 1909 and 1987.

 It contests the Basque derby against rivals Athletic Bilbao.

 Real Sociedad were founder members of La Liga in 1928, and its longest spell in the top flight was for 40 seasons, from 1967 to 2007.
</item 8>

<item 9>
Graph query row:
person: Darko Kovačević
connection: ['FROM_CHUNK', 'FROM_CHUNK']
connected_person: Nihat Kahveci
nationality: Turkish
</item 9>

<item 10>
Nihat Kahveci
Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
</item 10>
=== END ITEM id=b1c306df0afc5dda ===

=== ITEM id=8c6a7855dc0add0a ===
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
L.M. Montgomery's Anne of Green Gables
L.M. Montgomery's Anne of Green Gables is a Canadian television film based on Lucy Maud Montgomery's novel of the same name.

 It first aired on YTV on February 15, 2016 and starred Ella Ballentine, Martin Sheen and Sara Botsford.

 Montgomery's granddaughter, Kate Macdonald Butler, was one of the film's executive producers.

 The film's world premiere was held February 2, 2016 at the Canadian Museum of History.
</item 1>

<item 2>
Emily's Quest
Emily's Quest is a novel and the last of the "Emily" trilogy by Lucy Maud Montgomery.

 After finishing "Emily Climbs", Montgomery suspended writing "Emily's Quest" and published "The Blue Castle"; she resumed writing and published in 1927.
</item 2>

<item 3>
Anne Shirley
Anne Shirley is a fictional character introduced in the 1908 novel "Anne of Green Gables" by Lucy Maud Montgomery.

 Montgomery wrote in her journal that the idea for Anne's story came from relatives who, planning to adopt an orphaned boy, received a girl instead.

 Anne Shirley's appearance was inspired by a photograph which Montgomery clipped from the Metropolitan Magazine and kept, unaware of the model's identity as the
1900s Gibson Girl Evelyn Nesbit.
</item 3>

<item 4>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
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
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 6>

<item 7>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 7>

<item 8>
Rosario Castellanos
Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.

 Along with the other members of the Generation of 1950 (the poets who wrote following the Second World War, influenced by César Vallejo and others), she was
one of Mexico's most important literary voices in the last century.

 Throughout her life, she wrote eloquently about issues of cultural and gender oppression, and her work has influenced feminist theory and cultural studies.

 Though she died young, she opened the door of Mexican literature to women, and left a legacy that still resonates today.
</item 8>

<item 9>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 9>

<item 10>
Anne of Ingleside
Anne of Ingleside is a children's novel by Canadian author Lucy Maud Montgomery.

 It was first published in July 1939 by McClelland and Stewart (Toronto) and the Frederick A. Stokes Company (New York).

 It is the tenth of eleven books that feature the character of Anne Shirley, and Montgomery's final published novel.

 (Two novels that occur later in the "Anne" chronology were actually published years earlier.

 As well, the short story collection "The Blythes Are Quoted", written in 1941/42, but not published until 2009, concludes the Anne chronology.)
</item 10>
=== END ITEM id=8c6a7855dc0add0a ===

=== ITEM id=dd1cb233707ce697 ===
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
</item 1>

<item 2>
Hamdard University
Hamdard University (Urdu: ) is a private research university with campuses in Karachi and Islamabad, Pakistan.

 It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.

 Hamdard is one of the first and the oldest private institutions of higher education in Pakistan.

 In Karachi, Hamdard University is the largest private research university with a campus area of over 350 acres.
</item 2>

<item 3>
The text profiles diverse educational and related institutions worldwide. Highlights include Berat University (Albania), founded in 2009 and expanding to four
faculties; Rotman (Toronto), established in 1950; Auburn, Arizona, Debrecen, Copenhagen, Dalhousie and other research universities; specialist schools such as
Falmouth and Aalborghus; and private or state colleges across Ghana, Malaysia, Norway, Pakistan and elsewhere. It also covers Syracuse sororities, a
</item 3>

<item 4>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 4>

<item 5>
Santa Barbara City College
Santa Barbara City College (SBCC) is a two-year community college that opened in 1909 and is located on a 74 acre campus in the city of Santa Barbara,
California, USA.

 SBCC was named co-winner of the 2013 Aspen Institute Prize for Community College Excellence, earning it a #1 national ranking.

 The college was recognized for its quality and focus in four areas: facilitating underrepresented and minority student success, student learning outcomes,
degree completion and transfer rates, and labor market success in securing good jobs after college.

 Santa Barbara City College was established by the Santa Barbara High School District in 1909, making it one of the oldest community colleges in California.

 The college was discontinued shortly after World War I, and its work largely taken over by the Santa Barbara State Normal School, which became the Santa
Barbara State College, and later, the University of California, Santa Barbara.
</item 5>

<item 6>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 6>

<item 7>
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
</item 7>

<item 8>
Mission Santa Barbara
Mission Santa Barbara, also known as Santa Barbara Mission, is a Spanish mission founded by the Franciscan order near present-day Santa Barbara, California.

 It was founded by Padre Fermín Lasuén on December 4, 1786, the feast day of Saint Barbara, as the tenth mission for the religious conversion of the indigenous
local Chumash-Barbareño tribe of Native American people.

 The mission is the namesake of the city of Santa Barbara as well as of Santa Barbara County.
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
Santa Barbara High School
Santa Barbara Senior High School, "Home of the Dons," is situated on a 40 acre campus in Santa Barbara, California and is part of the Santa Barbara Unified
School District.

 One of the oldest high schools in California, it was established in 1875, and moved to its current site in 1924.

 Until the creation of two rival high schools in 1959 and 1966, it was the sole public high school serving the city of Santa Barbara.

 In November 2005, its 18th-century Spanish influenced design was named an official California and City historic landmark.

 Today, Santa Barbara High School has an enrollment of roughly 2,200 pupils and certified staff numbering 108.
</item 10>
=== END ITEM id=dd1cb233707ce697 ===

=== ITEM id=b448c4a5f38ebf32 ===
Gold sentences and facts:
Sentence 0: Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.
  Fact 0.0: Batman is a 1989 American superhero film.
  Fact 0.1: Batman was directed by Tim Burton.
  Fact 0.2: Batman was produced by Jon Peters and Peter Guber.
  Fact 0.3: Batman is based on the DC Comics character of the same name.
Sentence 1:  He is Chairman and CEO of Mandalay Entertainment.
  Fact 1.0: The person referred to as 'he' is Chairman of Mandalay Entertainment.
  Fact 1.1: The person referred to as 'he' is CEO of Mandalay Entertainment.

Retrieved items:
<item 1>
Batman (1989 film)
Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.

 It is the first installment of Warner Bros.' initial "Batman" film series.

 The film stars Michael Keaton as Bruce Wayne/Batman and Jack Nicholson as The Joker, alongside Kim Basinger, Robert Wuhl, Pat Hingle, Billy Dee Williams,
Michael Gough and Jack Palance.

 The film takes place early in the title character's war on crime, and depicts a battle with his arch-nemesis the Joker.
</item 1>

<item 2>
Crazy for You (Madonna song)
"Crazy for You" is a song by American singer Madonna from the soundtrack album to the 1985 film "Vision Quest".

 It was released on March 2, 1985 by Geffen Records as the first single from the soundtrack.

 The song appears remixed on the greatest hits compilation "The Immaculate Collection" (1990) and was re-released on February 24, 1991 by Sire Records to
promote the album.

 The song was also included on the ballads compilation "Something to Remember" (1995) and the greatest hits compilation "Celebration" (2009).

 Producers Jon Peters and Peter Guber, along with music director Phil Ramone, decided to use Madonna after listening to her previous recordings, employing John
Bettis and Jon Lind to write the song.

 After reading the script of the film, Bettis and Lind wrote the song about the situation in which the lead characters meet at a nightclub.

 Initial recording sessions did not impress Bettis and Lind, and they felt that "Crazy for You" would be dropped from the soundtrack.

 However, a new version was recorded to their liking.
</item 2>

<item 3>
The Man Who Falls
"The Man Who Falls" is a 1989 comic book story by Dennis O'Neil and Dick Giordano.

 It is an overview of Bruce Wayne's early life, including his parents' murder, his time spent traveling and training throughout the world, and his return to
Gotham City to become Batman.

 Sixteen years later, the story became the structural basis for "Batman Begins", which rebooted the Batman film franchise in 2005.
</item 3>

<item 4>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 4>

<item 5>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 5>

<item 6>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 6>

<item 7>
God Loves, Man Kills (album)
God Loves, Man Kills was One King Down's second release but first full-length album.

 It was released on December 8, 1998, through Equal Vision Records.

 Featuring new singer Jon Peters (who would eventually be replaced by original One King Down vocalist, Rob Fusco, this record saw One King Down attempt to merge
funk and skiffle into their sound.

 The results received mixed reactions.
</item 7>

<item 8>
Batman: Return of the Joker
Batman: Return of the Joker, known in Japan as Dynamite Batman (ダイナマイトバットマン , Dainamaito Battoman ) , is a 1991 platform video game, the follow-up to Sunsoft's
on the Nintendo Entertainment System.

 Unlike that game, which was based on the 1989 Batman film directed by Tim Burton, "Return of the Joker" is entirely self-contained and based more on the modern
comic book iteration of Batman.

 However, Batman mans the Batmobile from the 1989 film.

 A remake of "Return of the Joker", titled Batman: Revenge of the Joker, was released on the Sega Genesis by Ringler Studios in 1992.

 A Super NES version of "Revenge of the Joker" was in development, but never released.
</item 8>

<item 9>
Batman vs. Two-Face
Batman vs. Two-Face is an upcoming 2017 American animated direct-to-video superhero film produced by Warner Bros.

 Animation and distributed by Warner Bros.

 Home Entertainment.

 It is a sequel to "".

 It will premiere at the New York Comic Con on October 8, 2017 and will be followed by a digital release on October 10, 2017 and on DVD and Blu-ray on October
17, 2017.

 Based on the 1960s "Batman" TV series, the film stars Adam West (in his final role before his death), Burt Ward and Julie Newmar reprising their roles of
Batman, Robin and Catwoman from the series.

 The film will show a tribute to the late Adam West.

 This could also be the final time Warner Bros.

 Animation would make a 1960s Batman film.
</item 9>

<item 10>
Peter Guber
Howard Peter Guber (born March 1, 1942) is an executive, entrepreneur, educator, and author.

 He is Chairman and CEO of Mandalay Entertainment.

 Guber's most recent films from Mandalay Entertainment include "The Kids Are All Right", "Soul Surfer", and "Bernie".

 He has also produced "Batman", "The Witches of Eastwick", and "Flashdance."

 Guber's films have earned over $3 billion worldwide and 50 Academy Award nominations.
</item 10>
=== END ITEM id=b448c4a5f38ebf32 ===

=== ITEM id=c14091156560c9b8 ===
Gold sentences and facts:
Sentence 0:  The event was broadcast live on BBC One and was hosted by Graham Norton.
  Fact 0.0: The event was broadcast live on BBC One.
  Fact 0.1: Graham Norton hosted the event.
Sentence 1: Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.
  Fact 1.0: Graham William Walker was born on 4 April 1963.
  Fact 1.1: Graham William Walker is better known as Graham Norton.
  Fact 1.2: Graham Norton is an Irish television presenter.
  Fact 1.3: Graham Norton is an Irish radio presenter.
  Fact 1.4: Graham Norton is an Irish DJ.
  Fact 1.5: Graham Norton is an Irish comedian.
  Fact 1.6: Graham Norton is an Irish actor.
  Fact 1.7: Graham Norton is an Irish writer.

Retrieved items:
<item 1>
British Academy Television Awards
The British Academy Television Awards, also known as the BAFTA Television Awards, are presented in an annual award show hosted by the British Academy of Film
and Television Arts (BAFTA).

 They have been awarded annually since 1955.

 It is the British equivalent to the Emmy Awards in the United States and Logie Awards in Australia.
</item 1>

<item 2>
British Academy Television Awards 1997
The 1997 British Academy Television Awards were held on 29 April at the Royal Albert Hall in London, as a joint ceremony with the British Academy Film Awards.

 To date, it is the last occasion upon which the two sets awards have been given jointly at the same event; from 1998, the Film and Television Awards were given
at separate ceremonies, for the first time since 1968.
</item 2>

<item 3>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 3>

<item 4>
British Academy Children's Awards
The British Academy Children's Awards are presented in an annual award show hosted by the British Academy of Film and Television Arts (BAFTA).

 They have been awarded annually since 1996.

 Before that, children's awards were a part of the main British Academy Television Awards.
</item 4>

<item 5>
The text surveys awards and broadcasters: the Soap Opera Digest Hottest Male Star award ran from 1993–1999. BAFTA Television Awards began in 1955; Best Actress
became performance-based in 1969, nominees were listed from 1970, and the category split in 2010. It notes ceremony details for 1997–2009 and 2012, plus the
separate Children’s Awards (since 1996) and BAFTA Scotland (formed 1986
</item 5>

<item 6>
Graham Norton
Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.

 He is a five-time BAFTA TV Award winner for his comedy chat show "The Graham Norton Show".

 Previously shown on BBC Two, it took the prestigious Friday night slot on BBC One from "The Jonathan Ross Show" in 2010.
</item 6>

<item 7>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 7>

<item 8>
British Academy Television Award for Best Actress
This is a list of the British Academy Television Awards for Best Actress.

 The British Academy Television Awards began in 1955.

 The Best Actress award was initially given as an 'individual honour' without credit to a particular performance until 1969 when Wendy Craig won for her
performance in "Not in Front of the Children".

 Since 1970, nominees have been announced in addition to the winner and are listed with the winner highlighted in blue.

 The Actress category was split into Leading Actress and Supporting Actress starting in 2010.
</item 8>

<item 9>
British Academy Television Awards 1998
The 1998 British Academy Television Awards were held on 18 May at the Prince of Wales Theatre in London.

 The ceremony was broadcast on ITV, hosted by Bob Monkhouse, and it was the first occasion since 1968 that the Television Awards had been held separately from
the British Academy Film Awards, instead of as a joint ceremony.
</item 9>

<item 10>
BAFTA Scotland
BAFTA in Scotland is the Scottish branch of the British Academy of Film and Television Arts.

 Formed in 1986, the branch holds two annual awards ceremonies recognising the achievement by performers and production staff in Scottish film, television and
video games.

 These Awards are separate from the British Academy Television Awards and British Academy Film Awards.
</item 10>
=== END ITEM id=c14091156560c9b8 ===

=== ITEM id=178632ec49b0fe98 ===
Gold sentences and facts:
Sentence 0:  Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
  Fact 0.0: Programming is simulcast on WANH (91.5 FM).
  Fact 0.1: WANH (91.5 FM) is located in Meredith.
  Fact 0.2: WANH serves the Lakes Region.
Sentence 1:  The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford County and Merrimack County.
  Fact 1.0: The area comprises all of Belknap County.
  Fact 1.1: The area comprises the southern portion of Carroll County.
  Fact 1.2: The area comprises the eastern portion of Grafton County.
  Fact 1.3: The area comprises the northern portion of Strafford County.
  Fact 1.4: The area comprises the northern portion of Merrimack County.

Retrieved items:
<item 1>
KRBT
KRBT (1340 AM) is a radio station broadcasting a sports format.

 Licensed to Eveleth, Minnesota, United States, the station serves the Iron Range area.

 The station is currently owned by Range Broadcasting, Inc., and features programming from the Red Zone Radio Network.

 Originally a stand alone music station with call letters WEVE, then a simulcast music station with WEVE-FM, the station changed it's call letters to KRBT for
"Range's Best Talk" in June of 1998 and discontinued simulcast with WEVE-FM to become a talk station.

 The format change to sports/talk was made following the death of previous owner, Lew Latto, Iron Range Broadcasting, Inc.

 President on August, 24 2011 and subsequent sale to Red Rock Radio in March 2012, and has been retained by the new owner as of July 2017, Range Broadcasting,
Inc.
</item 1>

<item 2>
KKBN
KKBN (93.5 FM) The Cabin a radio station since 1986 broadcasting a Country music format since March 2000.

 Licensed to Twain Harte, California, United States, the station serves the greater Mother Lode area comprising Tuolumne and Calaveras Counties with a combined
population of approximately 110,000.

 The station is currently owned by Clarke Broadcasting Corporation.
</item 2>

<item 3>
KMBC-TV
KMBC-TV, virtual channel 9 (UHF digital channel 29), is the ABC-affiliated television station licensed to Kansas City, Missouri, United States and also serving
Kansas City, Kansas.

 The station is owned by the Hearst Television subsidiary of the Hearst Corporation, as part of a duopoly with CW affiliate KCWE (channel 29).

 The two stations share studio facilities located at the Winchester Business Center on Winchester Avenue and East 63rd Street (near Swope Park, off of I-435) in
southeastern Kansas City, Missouri; KMBC-TV maintains transmitter facilities located near the Blue River in eastern Kansas City.

 On cable, KMBC-TV is available on Charter Spectrum, Comcast Xfinity and Consolidated Communications channel 12, and Google Fiber and AT&T U-verse channel 9.

 There is a high definition feed provided on Spectrum digital channel 1200, Xfinity channel 812, Consolidated channel 610 and U-verse channel 1009.
</item 3>

<item 4>
KLIK
KLIK (1240 AM), branding as Newstalk 1240, is a radio station broadcasting a News Talk Information format.

 Licensed to Jefferson City, Missouri, United States, the station serves the Columbia, Missouri area.

 The station is currently owned by Cumulus Media and features programing from ABC Radio and Westwood One.

 KLIK also operates a local news operation with sister station KFRU (1400 AM in Columbia, Missouri).

 From 1954 until September 8, 1999, KLIK was located at 950 AM, transmitting with a daytime power of 5000 watts and a nighttime power of 500 watts (directional)
from a four tower array about 3.2 miles south of Jefferson City.

 Early owners of KLIK broadcast a varied format of news and talk programs including music programs of middle of the road, top 40, adult contemporary and country
music as 95 KLIK.

 For many years, KLIK and KJFF as the two largest regional radio stations (the most powerful AM and FM station in the region) dominated radio listenership in
cumulative market share in the Columbia-Jeff City Market of Central Missouri.
</item 4>

<item 5>
WLMR
WLMR (1450 AM) is a radio station broadcasting a religious format.

 Licensed to Chattanooga, Tennessee, USA, the station serves the Chattanooga area.

 The station is currently owned by Wilkins Communications Network, Inc. and features programming from USA Radio Network.

 In the early 1980s, the station was automated and played country music from studios in a strip mall on Brainerd Rd.

 It was also Chattanooga's first talk radio station when it held the WZRA call letters.

 WZRA was the first home of Jeff Styles, and also featured well known personalities such as Kelly McCoy and Robert T. Nash.
</item 5>

<item 6>
KWXL-LP
KWXL-LP (98.7 FM) is a high school radio station broadcasting a variety format.

 Licensed to Tucson, Arizona, United States, the station serves the Tucson area.

 The station is currently owned by the Tucson Unified School District.

 KWXL-LP is Tucson's only high school radio station.

 It is also a news radio station for students who attend Pueblo High Magnet School.

 Students broadcasting over the radio station are enrolled in a "Writing/Reporting for Broadcasting" class with instructor Sarah Walson.

 Originally started by Douglas Potter, who retired in 2006.

 Listeners can also find FM 98.7, KWXL on iTunes.
</item 6>

<item 7>
WVNH
WVNH (91.1 FM) is a radio station broadcasting a Christian radio format.

 Licensed to Concord, New Hampshire, United States, the station serves the Concord area.

 The station is currently owned by New Hampshire Gospel Radio, Inc.

 Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
</item 7>

<item 8>
KVKI-FM
KVKI-FM (96.5 FM) is a radio station broadcasting an adult contemporary music format.

 Licensed to Shreveport, Louisiana, United States, the station serves the Shreveport area.

 The station is currently owned by Townsquare Media.

 Its studios are shared with its other five sister stations in West Shreveport (one mile west of Shreveport Regional Airport), and the transmitter is in
Blanchard, Louisiana.

 The call letters were originally KBCL, and later became religious station KEPT, before the call letters were changed to KVKI to be similar to the call letters
of the popular KVIL radio station in Dallas, Texas.
</item 8>

<item 9>
KXMB-TV
KXMB-TV, channel 12, is the CBS-affiliated television station for Bismarck, North Dakota.

 It broadcasts a high definition digital signal on VHF channel 12 (or virtual channel 12.1 via PSIP) from a transmitter near St. Anthony.

 The station can also be seen on Midcontinent cable channel 11 in Bismarck-Mandan, and cable channel 12 in most other areas.

 There is a high definition feed provided on Midcontinent digital channel 611.
</item 9>

<item 10>
Lakes Region (New Hampshire)
The Lakes Region of New Hampshire is the mid-state region surrounding Lake Winnipesaukee, Winnisquam Lake, Squam Lake, and Newfound Lake.

 The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford
County and Merrimack County.

 The largest municipality is the city of Laconia.
</item 10>
=== END ITEM id=178632ec49b0fe98 ===

=== ITEM id=cf184a2ec231dac5 ===
Gold sentences and facts:
Sentence 0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
  Fact 0.0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book.
  Fact 0.1: Lawrence Weschler wrote Mr. Wilson's Cabinet of Wonder.
  Fact 0.2: Mr. Wilson's Cabinet of Wonder is primarily about the Museum of Jurassic Technology.
  Fact 0.3: Mr. Wilson's Cabinet of Wonder is more broadly about the history and role of museums.
Sentence 1: The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.
  Fact 1.0: The Museum of Jurassic Technology is a museum.
  Fact 1.1: The Museum of Jurassic Technology is located at 9341 Venice Boulevard.
  Fact 1.2: The Museum of Jurassic Technology is in the Palms district of Los Angeles, California.

Retrieved items:
<item 1>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 1>

<item 2>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 2>

<item 3>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 3>

<item 4>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 4>

<item 5>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 5>

<item 6>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 6>

<item 7>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 7>

<item 8>
Graph query row:
museum: Museum of Jurassic Technology
connection: ['FROM_CHUNK', 'FROM_CHUNK']
work: Mr. Wilson's Cabinet of Wonder
city: Los Angeles
</item 8>

<item 9>
John G. Wilson
John Guy Wilson (1842–1892) was a Manchester, UK patent agent and the founder of Wilson Gunn, one of the oldest firms of Patent and Trade Mark Attorneys in the
UK.

 John G. Wilson built up the business until his death in the 1890s.

 As a side line he acted as agent to the Alliance Insurance Company.

 There is no record that Mr. Wilson ever became a member of the Chartered Institute of Patent Agents.
</item 9>

<item 10>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 10>
=== END ITEM id=cf184a2ec231dac5 ===

=== ITEM id=2fb85b7575495807 ===
Gold sentences and facts:
Sentence 0: of Montreal is an American experimental pop band from Athens, Georgia.
  Fact 0.0: of Montreal is an American band.
  Fact 0.1: of Montreal is an experimental pop band.
  Fact 0.2: of Montreal is from Athens, Georgia.
Sentence 1: Vib Gyor was a band that was formed in Leeds, England.
  Fact 1.0: Vib Gyor was a band.
  Fact 1.1: Vib Gyor was formed in Leeds, England.

Retrieved items:
<item 1>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 1>

<item 2>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 2>

<item 3>
Fallen (Vib Gyor song)
"Fallen" was the first single from the Leeds band Vib Gyor.

 It was first given limited release in July 2006, and has attained critical acclaim and a large amount of radio play on both sides of the atlantic.
</item 3>

<item 4>
The White EP (Vib Gyor EP)
The "White EP" is a promo EP from the Leeds band Vib Gyor.

 It was first sold on their UK tour with American band Transfer and was released through their My Space page and official web-site on 13 November 2007.

 It is the follow-up to their The Secret EP.

 It was recorded in Rockfield studios in South Wales and produced by Ken Thomas.
</item 4>

<item 5>
Bloc Party
Bloc Party are an English indie rock band, currently composed of Kele Okereke (lead vocals, rhythm guitar, keyboards, sampler), Russell Lissack (lead guitar,
keyboards), Justin Harris (bass guitar, keyboards, saxophones, backing vocals) and Louise Bartle (drums, percussion).

 Former members Matt Tong and Gordon Moakes left the band in 2013 and 2015 respectively.

 Their brand of music, whilst rooted in rock, retains elements of other genres such as electronica and house music.

 The band was formed at the 1999 Reading Festival by Okereke and Lissack.

 They went through a variety of names before settling on Bloc Party in 2003.

 Moakes joined the band after answering an advert in "NME" magazine, while Tong was picked via an audition.

 Bloc Party got their break by giving BBC Radio 1 DJ Steve Lamacq and Franz Ferdinand's lead singer, Alex Kapranos, a copy of their demo "She's Hearing Voices".
</item 5>

<item 6>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 6>

<item 7>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 7>

<item 8>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 8>

<item 9>
Electric Guitars
Electric Guitars were an English band formed early in 1980 by Neil Davenport (vocals, lyrics) and Richard Hall (bass, vocals) who were both studying English at
Bristol University.

 The band soon increased to a five-man line-up, with Andy Saunders (guitar, vocals), Matt Salt (drums) and Dick Truscott (keyboards), they also later added two
backing singers: Sara and Wendy Patridge.

 Their first single "Health" / "Continental Shelf" was released on local label Fried Egg Records in 1980.
</item 9>

<item 10>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 10>
=== END ITEM id=2fb85b7575495807 ===

=== ITEM id=cc998f859a5fc7bd ===
Gold sentences and facts:
Sentence 0: During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.
  Fact 0.0: Operation Cold Comfort was a failed SAS raid.
  Fact 0.1: Operation Cold Comfort took place during World War II.
  Fact 0.2: Operation Cold Comfort began with a parachute drop.
  Fact 0.3: The parachute drop of Operation Cold Comfort was north of Verona.
  Fact 0.4: The parachute drop of Operation Cold Comfort was on February 17, 1945.
Sentence 1:  The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.
  Fact 1.0: The SAS was founded in 1941 as a regiment.
  Fact 1.1: The SAS was later reconstituted as a corps.
  Fact 1.2: The SAS was reconstituted as a corps in 1950.

Retrieved items:
<item 1>
Special Operations Force (Singapore)
The Special Operations Force (SOF) is the Republic's Army Special Forces composed of highly trained elite soldiers within the Singapore Armed Forces Commando
Formation and an essential component of the joint special forces unit Special Operations Task Force (SOTF).

 According to the Principles of Special Forces, the Republic's special forces thrive on their exceptional qualities and advanced skills, consisting highly
adaptive individuals who are independent and can operate independently, and the Republic's special forces soldiers cannot be mass-produced and must be managed
carefully.
</item 1>

<item 2>
Special Air Service
The Special Air Service (SAS) is a special forces unit of the British Army.

 The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.

 The unit undertakes a number of roles including covert reconnaissance, counter-terrorism, direct action and hostage rescue.
</item 2>

<item 3>
Operation Cold Comfort
During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.

 It was later renamed Zombie.
</item 3>

<item 4>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 4>

<item 5>
Operation Barras
Operation Barras was a British Army operation that took place in Sierra Leone on 10 September 2000, during the late stages of that nation's civil war.

 The operation aimed to release five British soldiers of the Royal Irish Regiment who had been held by a militia group known as the "West Side Boys".

 The soldiers were part of a patrol that was returning from a visit to Jordanian peacekeepers attached to the United Nations Mission in Sierra Leone (UNAMSIL)
at Masiaka on 25 August 2000 when they turned off the main road and down a track towards the village of Magbeni.

 There the patrol was overwhelmed by a large number of heavily armed rebels, taken prisoner, and transported to Gberi Bana on the opposite side of Rokel Creek.
</item 5>

<item 6>
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
</item 6>

<item 7>
Jan Breytenbach
Jan Dirk Breytenbach {'1': ", '2': ", '3': ", '4': "} (born 4 July 1933) was appointed by General Fritz Loots, the founder of the South African Special Forces
Brigade, as the first commander of 1 Reconnaissance Commando, the first unit founded within the South African Special Forces.

 He was also appointed as the first commander of the 32 Battalion, known colloquially as "Buffalo Battalion", as well as 44 Parachute Brigade.
</item 7>

<item 8>
Simultaneous action selection
Simultaneous action selection, or SAS, is a game mechanic that occurs when players of a game take action (such as moving their pieces) at the same time.

 An example of a game that uses this type of movement is the game Diplomacy.

 Typically, a "secret yet binding" method of committing to one's move is necessary, so that as players' moves are revealed and implemented, others do not change
their moves in light of the new information.

 Thus, in Diplomacy, players write down their moves and then reveal them simultaneously.

 Because no player gets the first move, this potentially arbitrary source of advantage is not present.

 It is also possible for simultaneous movement games to proceed relatively quickly, because players are acting at the same time, rather than waiting for their
turn.

 Simultaneous action selection is easily implemented in card games such as Apples to Apples in which players simply select cards and throw them face-down into
the center.
</item 8>

<item 9>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 9>

<item 10>
Joint Special Operations Command (Jordan)
The Special Operation Forces of the Jordanian Armed Forces serve as Jordan's premiere special forces unit.

 Founded on April 15, 1963 on the orders of the late King Hussein, its primary roles include reconnaissance, counter-terrorism, search and evacuation,
intelligence gathering combat, and the protection of key sites.

 The Special Operation Forces are also charged with carrying out precision strikes against critical enemy targets.

 The 14,000-strong unit are equipped and trained to be able to operate behind enemy lines for long periods without any logistical support, and is considered one
of the finest special forces units in the world.
</item 10>
=== END ITEM id=cc998f859a5fc7bd ===

=== ITEM id=13c546bfdab944a0 ===
Gold sentences and facts:
Sentence 0: The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).
  Fact 0.0: The Washington City Canal operated from 1815 until the mid-1850s.
  Fact 0.1: The Washington City Canal operated in Washington, D.C.
  Fact 0.2: The Washington City Canal connected the Anacostia River to Tiber Creek.
  Fact 0.3: The Washington City Canal connected the Anacostia River to the Potomac River.
  Fact 0.4: The Washington City Canal later connected the Anacostia River to the Chesapeake and Ohio Canal (C&O).
  Fact 0.5: The Anacostia River was called the 'Eastern Branch' at that time.
Sentence 1: The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River to Hagerstown, Indiana.
  Fact 1.0: The Whitewater Canal was built between 1836 and 1847.
  Fact 1.1: The Whitewater Canal spanned a distance of seventy-six miles.
  Fact 1.2: The Whitewater Canal stretched from Lawrenceburg, Indiana.
  Fact 1.3: Lawrenceburg, Indiana is on the Ohio River.
  Fact 1.4: The Whitewater Canal stretched to Hagerstown, Indiana.

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
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 2>

<item 3>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 3>

<item 4>
Whitewater River (Great Miami River)
The Whitewater River is a 101 mi southerly flowing right tributary of the Great Miami River in southeastern Indiana and southwestern Ohio in the United States.

 It is formed by the confluence of two forks, the West Fork and East Fork.

 The name is a misnomer, as there is no true white water on the river.

 However, there are many rapids due to the steep gradient present - the river falls an average of 6 ft/mile .

 The gradient rendered upstream navigation impossible, and in the mid-nineteenth century resulted in the construction of the Whitewater Canal paralleling the
river from north of Connersville, Indiana, to the Ohio River.
</item 4>

<item 5>
Tiber Creek
Tiber Creek or Tyber Creek is a tributary of the Potomac River in Washington, D.C. In the 19th century it was modified to become part of the Washington City
Canal, and subsequently was enclosed in a tunnel.
</item 5>

<item 6>
Duck Creek Aqueduct
Duck Creek Aqueduct, also known as the Metamora Aqueduct and Whitewater Canal Aqueduct, is a historic aqueduct carrying the Whitewater Canal over Duck Creek in
Metamora Township, Franklin County, Indiana.

 It is the only surviving covered wood aqueduct in the United States as such it is a form of covered bridge.

 It is a single-span Burr through truss aqueduct and measures approximately 90 feet long, 25 feet wide, and 25 feet deep.

 The original aqueduct was built between 1839 and 1843.

 The present structure was built in 1846, after the original aqueduct was washed out in a flood.

 It was strengthened in 1868, and repaired in 1901.

 After abandonment and deterioration, the Duck Creek Aqueduct was restored to its present appearance in 1946-1949.
</item 6>

<item 7>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 7>

<item 8>
Whitewater Canal
The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River
to Hagerstown, Indiana.
</item 8>

<item 9>
Washington City Canal
The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at
that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).

 The canal fell into disuse in the late 19th century and the city government covered over or filled in various sections.
</item 9>

<item 10>
Cincinnati and Whitewater Canal Tunnel
The Cincinnati and Whitewater Canal Tunnel is a historic yet abandoned canal tunnel in the southwestern corner of the U.S. state of Ohio.

 Located within the village of Cleves near Cincinnati, it was constructed in 1837 as part of the Whitewater Canal system.

 Since the canal's closure, it has largely been forgotten, but it has been designated a historic site.
</item 10>
=== END ITEM id=13c546bfdab944a0 ===
