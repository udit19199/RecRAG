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

=== ITEM id=82a73d202295860b ===
Gold sentences and facts:
Sentence 0:  At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
  Fact 0.0: She won a silver medal at the 2002 Commonwealth Games.
  Fact 0.1: She won the silver medal with Sara Petersen in the women's doubles at the 2002 Commonwealth Games.
  Fact 0.2: She won a bronze medal in the mixed team at the 2002 Commonwealth Games.
Sentence 1: The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25 July to 4 August 2002.
  Fact 1.0: The 2002 Commonwealth Games were officially known as the XVII Commonwealth Games.
  Fact 1.1: The 2002 Commonwealth Games were commonly known as Manchester 2002.
  Fact 1.2: The 2002 Commonwealth Games were held in Manchester, England.
  Fact 1.3: The 2002 Commonwealth Games were held from 25 July to 4 August 2002.

Retrieved items:
<item 1>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 1>

<item 2>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 2>

<item 3>
Nicole Gordon (badminton)
Nicole Gordon (born 17 March 1976 in Lower Hutt, New Zealand) is a female badminton player from New Zealand.

 At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
</item 3>

<item 4>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 4>

<item 5>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 5>

<item 6>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 6>

<item 7>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 7>

<item 8>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 8>

<item 9>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 9>

<item 10>
2001 Goodwill Games
The 2001 Goodwill Games was the fifth and final edition of the international multi-sport event.

 The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.

 A total around 1300 athletes took part in 14 sporting competitions.
</item 10>
=== END ITEM id=82a73d202295860b ===

=== ITEM id=e9d2fd45efd74968 ===
Gold sentences and facts:
Sentence 0:  His siblings were Abby, Nelson, Laurance, Winthrop, and David.
  Fact 0.0: Abby was his sibling.
  Fact 0.1: Nelson was his sibling.
  Fact 0.2: Laurance was his sibling.
  Fact 0.3: Winthrop was his sibling.
  Fact 0.4: David was his sibling.
Sentence 1: David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.
  Fact 1.0: David Rockefeller lived from June 12, 1915 to March 20, 2017.
  Fact 1.1: David Rockefeller was an American banker.
  Fact 1.2: David Rockefeller was chairman of Chase Manhattan Corporation.
  Fact 1.3: David Rockefeller was chief executive of Chase Manhattan Corporation.

Retrieved items:
<item 1>
David Rockefeller
David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.

 He was the oldest living member of the Rockefeller family and family patriarch from August 2004 until his death in March 2017.

 Rockefeller was a son of John D. Rockefeller Jr. and Abby Aldrich Rockefeller, and a grandson of John D. Rockefeller and Laura Spelman Rockefeller.
</item 1>

<item 2>
Rockefeller Brothers Fund
The Rockefeller Brothers Fund (RBF) is a philanthropic foundation created and run by members of the Rockefeller family.

 It was founded in New York City in 1940 as the primary philanthropic vehicle for the five third-generation Rockefeller brothers: John D. Rockefeller III,
Nelson, Laurance, Winthrop and David.

 It is distinct from the Rockefeller Foundation.

 The Rockefellers are an industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the late 19th and
early 20th centuries.
</item 2>

<item 3>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 3>

<item 4>
JPMorgan Chase
JPMorgan Chase & Co. is a U.S. multinational banking and financial services holding company headquartered in New York City.

 It is the largest bank in the United States, the world's sixth largest bank by total assets, with total assets of US$2.5 trillion, and the world's second most
valuable bank by market capitalization, after ICBC.

 It is a major provider of financial services, and according to "Forbes" magazine is the world's sixth largest public company based upon a composite ranking.

 The hedge fund unit of JPMorgan Chase is the second largest hedge fund in the United States.

 The company was formed in 2000, when Chase Manhattan Corporation merged with J.P. Morgan & Co.
</item 4>

<item 5>
Hope Aldrich Rockefeller
Hope Aldrich Rockefeller (born May 17, 1938) is an American newspaper publisher.

 She is the eldest daughter of philanthropist John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992).

 Her elder brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).
</item 5>

<item 6>
John D. Rockefeller III
John Davison Rockefeller III (March 21, 1906 – July 10, 1978) was a philanthropist and third-generation member of the prominent Rockefeller family.

 He was the eldest son of philanthropists John D. Rockefeller Jr. and Abby Aldrich Rockefeller.

 His siblings were Abby, Nelson, Laurance, Winthrop, and David.
</item 6>

<item 7>
Walter V. Shipley
Walter V. Shipley (born November 2, 1935) was the former Chairman and Chief Executive Officer of Chase Manhattan Bank and, before that, the company with which
it merged Chemical Bank.

 Shipley was named chief executive of Chemical in 1981 and held the position through 1999 and remained at the bank as chairman through January 2000, just prior
to the bank's merger with J.P. Morgan & Co..

 During his 18-year tenure, Shipley oversaw Chemical's mergers with Texas Commerce Bank in 1987, Manufacturers Hanover in 1991 and Chase Manhattan Bank in 1996.
</item 7>

<item 8>
William B. Harrison Jr.
William B. Harrison Jr., born August 12, 1943, in Rocky Mount, North Carolina, is the former CEO and chairman of JPMorgan Chase.

 He attended high school at Virginia Episcopal School, where he was a basketball star.

 He attended the University of North Carolina at Chapel Hill, where he was admitted to the Zeta Psi fraternity.

 Having risen through the ranks of Chemical Bank before succeeding Walter V. Shipley during the Chemical Bank takeover of the Chase Manhattan Corporation, he
and Douglas A. Warner III, then CEO of J.P. Morgan & Co., were the principal architects of the US$30.9 billion Chase and J.P. Morgan & Co. merger of 2000.

 Harrison has been a director of the Firm or a predecessor institution since 1991.

 Harrison is also a director of Merck & Co., Inc.
</item 8>

<item 9>
Chase Bank
JPMorgan Chase Bank, N.A., doing business as Chase Bank, is a national bank that constitutes the consumer and commercial banking subsidiary of the U.S.
multinational banking and financial services holding company, JPMorgan Chase.

 The bank was known as Chase Manhattan Bank until it merged with J.P. Morgan & Co. in 2000.

 Chase Manhattan Bank was formed by the merger of the Chase National Bank and The Manhattan Company in 1955.

 The bank has been headquartered in Columbus, Ohio since its merger with Bank One Corporation in 2004.

 The bank acquired the deposits and most assets of Washington Mutual.
</item 9>

<item 10>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 10>
=== END ITEM id=e9d2fd45efd74968 ===

=== ITEM id=cfdce31ef85728c8 ===
Gold sentences and facts:
Sentence 0: Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.
  Fact 0.0: Jerome John Garcia was born August 1, 1942.
  Fact 0.1: Jerome John Garcia died August 9, 1995.
  Fact 0.2: Jerome John Garcia was an American singer-songwriter.
  Fact 0.3: Jerome John Garcia was an American guitarist.
  Fact 0.4: Jerome John Garcia was best known for his work as the lead guitarist of the Grateful Dead.
  Fact 0.5: Jerome John Garcia was best known for his work as a vocalist with the Grateful Dead.
  Fact 0.6: The Grateful Dead came to prominence during the counterculture era in the 1960s.
Sentence 1: Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
  Fact 1.0: Joe Gooch was born 3 May 1977.
  Fact 1.1: Joe Gooch was Ten Years After's most recent lead vocalist.
  Fact 1.2: Joe Gooch was Ten Years After's most recent lead guitarist.

Retrieved items:
<item 1>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 1>

<item 2>
On Broadway: Act One – October 28th, 1987
On Broadway: Act One – October 28th, 1987 is a three-CD live album by the Jerry Garcia Band and the Jerry Garcia Acoustic Band, two music groups led by Grateful
Dead guitarist and singer Jerry Garcia.

 It contains three complete sets of music, recorded at the Lunt-Fontanne Theatre in New York City on October 28, 1987.

 It was released by ATO Records on June 23, 2015.
</item 2>

<item 3>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 3>

<item 4>
The Very Best of Jerry Garcia
The Very Best of Jerry Garcia is a compilation album with recordings by Jerry Garcia, Grateful Dead, Jerry Garcia Band, Old and in the Way, Jerry Garcia
Acoustic Band, and Reconstruction.

 It is meant to showcase Jerry's non-Grateful Dead work.
</item 4>

<item 5>
Garcia Plays Dylan
Garcia Plays Dylan is an album composed of various live performances featuring Jerry Garcia playing covers of Bob Dylan songs.

 It is culled from performances from 1973–1995, and features Garcia playing with Grateful Dead, Legion of Mary, Jerry Garcia Band, and Garcia-Saunders.

 Garcia takes lead vocals on all tracks.
</item 5>

<item 6>
Cats Under the Stars
Cats Under the Stars is an album by the Jerry Garcia Band.

 Released in 1978, it is the only studio album credited to the band.

 It was Jerry Garcia's first album on the Arista label and his first LP using the band name Jerry Garcia Band.

 It includes only one song that was performed by the Grateful Dead.

 Keith and Donna Godchaux, who were at the time members of the Garcia Band, contributed to the music.
</item 6>

<item 7>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 7>

<item 8>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 8>

<item 9>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 9>

<item 10>
Sugaree
"Sugaree" is a song written by long-time Grateful Dead lyricist Robert Hunter and composed by guitarist Jerry Garcia.

 It was written for Jerry Garcia's first solo album "Garcia", which was released in January 1972.

 As with the songs on the rest of the album, Garcia plays every instrument himself (except drums, played by Bill Kreutzmann), including acoustic guitar, bass
guitar, and an electric guitar played through a Leslie speaker.
</item 10>
=== END ITEM id=cfdce31ef85728c8 ===

=== ITEM id=7a4fcac51352a67b ===
Gold sentences and facts:
Sentence 0: Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that confirmed Trevi's status as a gay icon.
  Fact 0.0: Todos Me Miran ('Everyone looks at me') is a single from the Mexican artist Gloria Trevi.
  Fact 0.1: Todos Me Miran reached number 32 on Latin charts.
  Fact 0.2: Todos Me Miran became a club anthem.
  Fact 0.3: Todos Me Miran confirmed Trevi's status as a gay icon.
Sentence 1: Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
  Fact 1.0: Gloria Trevi was born Gloria de los Angeles Trevino Ruiz.
  Fact 1.1: Gloria Trevi was born on February 15, 1968.
  Fact 1.2: Gloria Trevi is a Mexican singer.
  Fact 1.3: Gloria Trevi is a Mexican songwriter.

Retrieved items:
<item 1>
La Trayectoria (Gloria Trevi album)
La Trayectoria (In English: The Trajectory) is a 2006 double live CD and multimedia DVD release by the Mexican singer Gloria Trevi.

 The DVD includes scenes from parts of the tour from the Palacio de los Deportes of the Mexico City and The Arena Monterrey and the CD includes four new songs
like: "Todos Me Miran" (They All Look At Me) is the first single compilation study of Mexican singer "Gloria Trevi", this is an unreleased song from the album.
</item 1>

<item 2>
A Cambio de Que
"A Cambio de Que" (In Exchange for What) is a song written by Xavier Santos Córtes and first performed by Mexican singer Marisela from her studio album
"Completamente Tuya".

 The song has been covered by several artists which have ranked on the music charts.

 Latin pop band Corvo covered the song in 1999 which reached #18 on the Hot Latin Songs chart in the United States.

 Mexican band Palomo's cover peaked at #31 on the Regional Mexican Airplay chart.

 American singer Huey Dunbar recorded "A Cambio de Que" on his debut studio album "Yo Si Me Enamore" (2001).

 This version peaked at #16 on the Tropical Songs chart.

 Mexican duranguense band Alacranes Musical covered the song on the album of the same name in 2004 which reached #25 on the Regional Mexican Airplay chart.
</item 2>

<item 3>
Mexico Espanol Airplay
Mexico Espanol Airplay is a record chart published weekly by "Billboard" magazine for Spanish language singles released in Mexico.

 According to "Billboard"' s electronic database, the first chart was published on October 1, 2011, with the track "Amor Clandestino" by Mexican band Maná at
number one; in the same year, fellow Mexican duo Jesse & Joy reached the top of the chart with their single "¡Corre!

", the first of their six number-one singles in the chart, the most for any act. "

¡Corre!"

 also won the Latin Grammy Awards for Record of the Year and Song of the Year in 2012.

 Starting on December 1, 2012, Mexican singer Thalía spent 10 non-consecutive weeks at number-one with "Manías", the first single from the album "Habítame
Siempre", which won the Lo Nuestro Award for Pop Album of the Year. "

Hoy Tengo Ganas de Ti" by Mexican artist Alejandro Fernández and American singer Christina Aguilera reached number-one in 2013, and also peaked at number four
in Spain.
</item 3>

<item 4>
De Contrabando
"De Contrabando" ("Smuggled") is a song written by Joan Sebastian.

 It was first performed by Mexican singer Imelda, and included on her album "Aparentemente", released in 1991.

 Six years later, was performed by fellow Mexican singer Zayda Y Los Culpables and released by Musart Records as the second single from her self-titled album in
1997.

 Mexican-American singer Jenni Rivera recorded a cover version for her studio album "Parrandera, Rebelde y Atrevida".

 "De Contrabando" was released as the second single from the album in 2006.

 In the United States, the song peaked at number fourteen on the "Billboard" Hot Latin Songs chart and number one on the "Billboard" Regional Mexican Songs
chart.

 The song was nominated for a Lo Nuestro Award for Regional Mexican Song of the Year at the 2007 Lo Nuestro Awards.

 Rivera was also recognized for Regional Mexican Female Artist of the Year.

 Rivera was also awarded Regional Mexican Airplay Song of the Year for at the 2007 Latin "Billboard" Music Awards.
</item 4>

<item 5>
Premio Lo Nuestro 2003
In the Regional Mexican, Pilar Montenegro took three awards for Regional Mexican Song of the Year, Pop Song of the Year ("Quitame Ese Hombre"), and for Best
Regional Mexican Female Artist.

 In the tropical genre, Celia Cruz took home with four great awards of the night for Best Salsa Performance, Best Tropical Female Artist, Best Tropical Song of
the year and Tropical Album of the Year.

 At the night, the greatest performance was a medley of top Latin hits from the last 15 years, performed by the artists that made them famous, including Vikki
Carr, Son by Four, Los Ilegales, La Mafia, Luis Enrique, Wilfrido Vargas and Olga Tañón.

 There was a great tribute to Celia Cruz by the world-famous salsa group "Fania All-Stars", of which Cruz was a member during the 1970s, reunited for an
exclusive performance that rocked the house.
</item 5>

<item 6>
Gloria Trevi
Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
</item 6>

<item 7>
Todos Me Miran
Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that
confirmed Trevi's status as a gay icon.

 The song, as interpreted in the music video, is about a young man who dares to crossdress in spite of society's opinions.
</item 7>

<item 8>
Tu Ángel de la Guarda
Tu Ángel de la Guarda ("Your guardian angel") is Gloria Trevi's second album, and it contained one of her signature songs and her most widely known hit, "Pelo
Suelto".

 It also contained other hit songs such as "Tu angel de la guarda", "Ya no", "Virgen de las virgenes", and "Hoy me ire de casa".

 This album was very controversial different from other artists' albums such as Lucero.

 "Virgen de las virgenes" mocked girls who said they were virgins but had actually lost their virginity, and "¡Ya no!"

 went against the machismo movement of México.

 Following the release of the album, Gloria was working hard in promoting radio, television and print media.

 Trevi first traveled abroad visiting American Union countries, Puerto Rico, Argentina, Colombia, Guatemala, Venezuela, Costa Rica, and received a Gold and
Platinum for high sales achieved.
</item 8>

<item 9>
Todos Los Romances
Todos Los Romances (English: "All the Romances" ) is a three-disc compilation album by Mexican singer Luis Miguel.

 Released on 11 August 1998 by WEA Latina, the record features the three previously released "Romance"-themed albums in which Miguel covered classic boleros in
each of them: "Romance" (1991), "Segundo Romance" (1994), and "Romances" (1997).

 An editor for AllMusic rated the album four of five stars.

 Commercially, "Todos Los Romances" peaked at number four in Spain and was certified double Platinum in the country.

 It also achieved Gold status in Argentina and peaked at number 12 on the "Billboard"'s Top Latin Albums in the United States.
</item 9>

<item 10>
No eres tú, soy yo
No eres tú, soy yo (English: It's not you, it's me ) is a 2010 Mexican romantic comedy directed by Alejandro Springall starring Eugenio Derbez, Alejandra Barros
and Martina García and based on the Argentinean film "No sos vos, soy yo" (2004), written by Juan Taratuto.

 It was produced by Matthias Eherenberg and filmed in Mexico City.
</item 10>
=== END ITEM id=7a4fcac51352a67b ===

=== ITEM id=fed1ce22223bed6c ===
Gold sentences and facts:
Sentence 0: Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of Pinocchio" by Carlo Collodi.
  Fact 0.0: Pinocchio is a 1940 American film.
  Fact 0.1: Pinocchio is an animated musical fantasy film.
  Fact 0.2: Pinocchio was produced by Walt Disney Productions.
  Fact 0.3: Pinocchio is based on the Italian children's novel 'The Adventures of Pinocchio'.
  Fact 0.4: The novel 'The Adventures of Pinocchio' was written by Carlo Collodi.
Sentence 1:  It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
  Fact 1.0: Pinocchio was the second animated feature film produced by Disney.
  Fact 1.1: Pinocchio was made after the success of 'Snow White and the Seven Dwarfs'.
  Fact 1.2: Snow White and the Seven Dwarfs is from 1937.
Sentence 2: Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.
  Fact 2.0: Home on the Range is a 2004 American film.
  Fact 2.1: Home on the Range is an animated musical western comedy film.
  Fact 2.2: Home on the Range was produced by Walt Disney Feature Animation.
  Fact 2.3: Home on the Range was released by Walt Disney Pictures.

Retrieved items:
<item 1>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 1>

<item 2>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
</item 2>

<item 3>
Pinocchio (1957 TV-musical)
Pinocchio, a 1957 television production of "Pinocchio", is a live (and live-action) musical version starring Mickey Rooney in the title role of the puppet who
wishes to become a real boy.

 Based on the novel by Carlo Collodi which also inspired the classic Walt Disney animated film, this version featured a now-forgotten new score by Alec Wilder
and William Engvick.

 It was telecast once on NBC as a television special, and, as far as is known, never rebroadcast by NBC, or even restaged with a different cast as was Rodgers
and Hammerstein's "Cinderella".

 Nor has it ever been issued on VHS or DVD.

 Other notable actors who appeared in the special included Walter Slezak (as Geppetto), Fran Allison (as the Blue Fairy), Martyn Green (as the Fox), Jerry
Colonna (as a Ringmaster), and Stubby Kaye as a Town Crier, a role he repeated in Wilder and Engvick's 1958 television musical, "Hansel and Gretel".

 "Pinocchio" was directed by noted Broadway choreographer Hanya Holm.
</item 3>

<item 4>
Pinocchio (1940 film)
Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of
Pinocchio" by Carlo Collodi.

 It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
</item 4>

<item 5>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 5>

<item 6>
Home on the Range (2004 film)
Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.

 The 45th Disney animated feature film, it was the last 2D animated Disney film released until "The Princess and the Frog" in 2009.

 Named after the popular country song of the same name, "Home on the Range" features the voices of Roseanne Barr, Judi Dench, Jennifer Tilly, Cuba Gooding Jr.,
Randy Quaid, and Steve Buscemi.

 The film is set in the Old West, and centers on a mismatched trio of dairy cows—brash, adventurous Maggie; prim, proper Mrs. Caloway; and ditzy, happy-go-lucky
Grace.

 The three cows must capture an infamous cattle rustler named Alameda Slim for his bounty in order to save their idyllic farm from foreclosure.

 Aiding them in their quest is Lucky Jack, a feisty, peg-legged rabbit, but a selfish horse named Buck, eagerly working in the service of Rico, a famous bounty
hunter, seeks the glory for himself.
</item 6>

<item 7>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 7>

<item 8>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 8>

<item 9>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 9>

<item 10>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 10>
=== END ITEM id=fed1ce22223bed6c ===

=== ITEM id=68c4bc1de0592036 ===
Gold sentences and facts:
Sentence 0: Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.
  Fact 0.0: Henderson Point is a census-designated place (CDP).
  Fact 0.1: Henderson Point is in Harrison County, Mississippi, United States.
Sentence 1:  Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
  Fact 1.0: Gulfport is a city adjoining Biloxi.
  Fact 1.1: Biloxi is a county seat of Harrison County.

Retrieved items:
<item 1>
Hendersonville, North Carolina
Hendersonville is a city in Henderson County, North Carolina United States.

 This city is 22 miles south of Asheville.

 It is the county seat of Henderson County.

 Like the county, the city is named for 19th-century North Carolina Supreme Court Chief Justice Leonard Henderson.
</item 1>

<item 2>
Henderson Point, Mississippi
Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.

 It is part of the Gulfport–Biloxi Metropolitan Statistical Area.

 The population was 170 at the 2010 census.
</item 2>

<item 3>
Raeford, North Carolina
Raeford is a city in Hoke County, North Carolina, United States.

 The population was 4,626 at the 2010 census.

 It is the county seat of Hoke County.

 The county was named after Confederate General Robert F. Hoke, a North Carolina native.
</item 3>

<item 4>
Paulding, Mississippi
Paulding is an unincorporated community in and one of the two county seats of Jasper County, Mississippi.

 It is the only unincorporated county seat in Mississippi.

 Settled in 1833, it was named by United States settlers in honor of Revolutionary War hero John Paulding.

 After its citizens refused to contribute to a new railroad, the community was bypassed in favor of Bay Springs, Mississippi, which was designated a railroad
stop to the west and the second county seat.

 It attracted more development and industry.
</item 4>

<item 5>
Henderson, Texas
Henderson is a city in Rusk County, northeast Texas, United States.

 The population was 13,712 at the 2010 census.

 It is the county seat of Rusk County.

 Henderson is named for James Pinckney Henderson, the first governor of Texas.
</item 5>

<item 6>
Nashville Terminal Subdivision
The Nashville Terminal Subdivision is a railroad line owned by CSX Transportation in the U.S. State of Tennessee.

 The Subdivision is broken up into 5 sections all in Nashville, Tennessee.

 The northern part of the Terminal is in Madison, Tennessee at milepost 000/0BA 174 on the southern end of the Ex-L&N Mainline Subdivision at Monfort.

 Disptach for the Mainline Sub is known as "LD" which is part of the Cincinnati Division.

 From here in Madison begins the double track that stays for another 22 miles south to Brentwood, Tennessee.

 At milepost 000/0BA/00H 176.6, the famous Johnny Cash "Amqui" location where the Ex-L&N Evansville, Indiana line, the Henderson Subdivision meets with the
Terminal.

 Dispatch for the Henderson Sub is known as "SA" and operates on AAR58.

 There are two crossings near each other, Williams Ave and Nesbitt Lane at Amqui.
</item 6>

<item 7>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 7>

<item 8>
Pensacola International Airport
Pensacola International Airport (IATA: PNS, ICAO: KPNS, FAA LID: PNS) , formerly Pensacola Gulf Coast Regional Airport and Pensacola Regional Airport (Hagler
Field), is a public use airport three nautical miles (6 km) northeast of the central business district of Pensacola, in Escambia County, Florida, United States.

 It is owned by the City of Pensacola.

 Despite the name, this airport does not offer direct international flights.

 This airport is one of the five major airports in North Florida, others being: Northwest Florida Beaches International Airport, Destin-Fort Walton Beach
Airport Tallahassee International Airport, and Jacksonville International Airport.
</item 8>

<item 9>
Piney Woods Country Life School
The Piney Woods Country Life School (or The Piney Woods School) is a co-educational independent historically African-American boarding school for grades 9-12 in
Piney Woods, unincorporated Rankin County, Mississippi.

 It is 21 mi south of Jackson.

 It is one of four remaining historically African-American boarding schools in the United States.

 It is currently the largest African-American boarding school, as well as being the second oldest continually operating African-American boarding school.
</item 9>

<item 10>
Biloxi, Mississippi
Biloxi ( ) is a city in Harrison County, Mississippi, United States.

 The 2010 United States Census recorded the population as 44,054, and in 2016 the estimated population was 45,975.

 Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
</item 10>
=== END ITEM id=68c4bc1de0592036 ===

=== ITEM id=0a3f52871ad02e3e ===
Gold sentences and facts:
Sentence 0: Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.
  Fact 0.0: Pagliacci is an Italian opera.
  Fact 0.1: Pagliacci has a prologue and two acts.
  Fact 0.2: Pagliacci has music and libretto by Ruggero Leoncavallo.
  Fact 0.3: The literal translation of Pagliacci is 'Clowns'.
Sentence 1: Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.
  Fact 1.0: Suor Angelica (Sister Angelica) is an opera.
  Fact 1.1: Suor Angelica is in one act.
  Fact 1.2: Suor Angelica is by Giacomo Puccini.
  Fact 1.3: Suor Angelica has an original Italian libretto by Giovacchino Forzano.

Retrieved items:
<item 1>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 1>

<item 2>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 2>

<item 3>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 3>

<item 4>
Giovacchino Forzano
Giovacchino Forzano (] ; 19 November 1884 – 28 October 1970) was an Italian playwright, librettist, stage director, and film director.

 A resourceful writer, he authored numerous popular plays and produced opera librettos for most of the major Italian composers of the early twentieth century,
including the librettos for Giacomo Puccini's "Suor Angelica" and "Gianni Schicchi".
</item 4>

<item 5>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 5>

<item 6>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 6>

<item 7>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 7>

<item 8>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 8>

<item 9>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 9>

<item 10>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 10>
=== END ITEM id=0a3f52871ad02e3e ===

=== ITEM id=f585a6ff68a099b5 ===
Gold sentences and facts:
Sentence 0: The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.
  Fact 0.0: The Sports Museum is also known as The Sports Museum of New England.
  Fact 0.1: The Sports Museum is a non-profit museum.
  Fact 0.2: The Sports Museum is currently located in the TD Garden.
  Fact 0.3: The TD Garden is in Boston, Massachusetts.
Sentence 1: TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.
  Fact 1.0: TD Garden is often called Boston Garden.
  Fact 1.1: TD Garden is often called "The Garden".
  Fact 1.2: TD Garden is a multi-purpose arena.
  Fact 1.3: TD Garden is in Boston, Massachusetts.
Sentence 2:  It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
  Fact 2.0: TD Garden opened in 1995.
  Fact 2.1: TD Garden opened as a replacement for the original Boston Garden.
  Fact 2.2: TD Garden has been known as Shawmut Center.
  Fact 2.3: TD Garden has been known as FleetCenter.
  Fact 2.4: TD Garden has been known as TD Banknorth Garden.

Retrieved items:
<item 1>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 1>

<item 2>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 2>

<item 3>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 3>

<item 4>
Alutiiq Museum
The Alutiiq Museum or Alutiiq Museum and Archaeological Repository is a non-profit museum and cultural center dedicated to preserving and sharing the cultural
traditions of the Koniag Alutiiq branch of Sugpiaq ~ Alutiiq of the Alaska Native people.

 The museum is a state-of-the-art facility and is located on the first floor of the "Alutiiq Center" of Kodiak, Alaska.

 Alutiiq Museum is one of three museums in Kodiak.

 The museum is the seventh museum in Alaska and the second "tribal museum" in the United States to be accredited.

 The museum provides tours of its exhibits, laboratory and collections storage facilities to educational groups.

 The museum will accept materials relevant to the prehistoric, historic, and contemporary cultural history of the Native peoples who settled the Koniag Alutiiq
Nation.

 Such materials include, but are not limited to, archaeological, ethnological, photographic, film, audio, archival, and natural history specimens.
</item 4>

<item 5>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 5>

<item 6>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 6>

<item 7>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 7>

<item 8>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 8>

<item 9>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 9>

<item 10>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 10>
=== END ITEM id=f585a6ff68a099b5 ===

=== ITEM id=5796efcd2093b2ab ===
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
1978 World Championship Tennis Finals
The 1978 World Championship Tennis Finals was a men's tennis tournament played on indoor carpet courts.

 It was the 8th edition of the WCT Finals and was part of the 1978 Colgate-Palmolive Grand Prix, as the World Championship Tennis and the Grand Prix circuits
were now combined.

 It was played at the Moody Coliseum in Dallas, Texas in the United States and was held from May 9 through May 14, 1978.

 Third-seeded Vitas Gerulaitis won the title and $100,000 first-prize money.
</item 1>

<item 2>
1979 World Championship Tennis Finals – Singles
Vitas Gerulaitis was the defending champion but lost in the semifinals to Björn Borg.
</item 2>

<item 3>
Heather Watson
Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.

 She is a former British No.1 and current British No. 2 behind Johanna Konta.
</item 3>

<item 4>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 4>

<item 5>
1977 Italian Open (tennis)
The 1977 Italian Open was a combined men's and women's tennis tournament that was played by men on outdoor clay courts at the Foro Italico in Rome, Italy.

 The men's tournament was part of the Colgate-Palmolive Grand Prix circuit while the women's tournament was part of the Colgate Series.

 The tournament was held from 16 May through 22 May 1977.

 The singles titles were won by eight-seeded Vitas Gerulaitis and fifth-seeded Janet Newberry who earned $21,000 and $6,000 first-prize money respectively.

 Gerulaitis competed despite being contracted to play for the Indiana Loves World Team Tennis (WTT) franchise and was fined $19,000 for failing to play Björn
Borg in the weekend of the final.
</item 5>

<item 6>
1974 Stadthalle Open – Singles
Vitas Gerulaitis won in the final 6–4, 3–6, 6–3, 6–2 against Andrew Pattison.
</item 6>

<item 7>
Vitas Gerulaitis
Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.

 In 1975, Gerulaitis won the men's doubles title at Wimbledon, partnering with Sandy Mayer.

 He won the men's singles title at one of the two Australian Open tournaments held in 1977 (Gerulaitis won the tournament that was held in December, while
Roscoe Tanner won the earlier January tournament).

 Gerulaitis also won two Italian Open titles, in 1977 and 1979, and the WCT Finals in Dallas, in 1978.
</item 7>

<item 8>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 8>

<item 9>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 9>

<item 10>
Jack Sock
Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis
Professionals (ATP).

 He is one of the top ranked American men in both singles and doubles on the ATP Tour.

 A former junior US Open champion, Sock's singles success is highlighted by 7 ATP finals, including three titles.

 His career best result came at the 2017 Indian Wells Masters, where he reached the semi-finals.
</item 10>
=== END ITEM id=5796efcd2093b2ab ===

=== ITEM id=9cd1013149933837 ===
Gold sentences and facts:
Sentence 0: Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).
  Fact 0.0: Dictyosperma is a monotypic genus of flowering plant.
  Fact 0.1: Dictyosperma is in the palm family.
  Fact 0.2: Dictyosperma is found in the Mascarene Islands in the Indian Ocean.
  Fact 0.3: Dictyosperma is found in Mauritius.
  Fact 0.4: Dictyosperma is found in Réunion.
  Fact 0.5: Dictyosperma is found in Rodrigues.
Sentence 1: The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in 1810.
  Fact 1.0: The genus Huernia is in the family Apocynaceae.
  Fact 1.1: The genus Huernia is in the subfamily Asclepiadoideae.
  Fact 1.2: The genus Huernia consists of stem succulents.
  Fact 1.3: The stem succulents of the genus Huernia are from Eastern Africa.
  Fact 1.4: The stem succulents of the genus Huernia are from Southern Africa.
  Fact 1.5: The genus Huernia was first described as a genus in 1810.

Retrieved items:
<item 1>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 1>

<item 2>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 2>

<item 3>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 3>

<item 4>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 4>

<item 5>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
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
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 7>

<item 8>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 8>

<item 9>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 9>

<item 10>
Harrimanella
Harrimanella is a genus of flowering plant in the heath family Ericaceae, with a single species, Harrimanella hypnoides, also known as moss bell heather.

 It was originally named Cassiope hypnoides by Linnaeus (1737) in his Flora Lapponica, but "Harrimanella hypnoides" is now the accepted name at ITIS.

 The species name "hypnoides" means 'like "Hypnum" ', which is a genus mosses.
</item 10>
=== END ITEM id=9cd1013149933837 ===

=== ITEM id=28c8e8dc8d930bcb ===
Gold sentences and facts:
Sentence 0: Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban parents in France, where she was also raised.
  Fact 0.0: Anaïs Nin was born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell.
  Fact 0.1: Anaïs Nin was born on February 21, 1903.
  Fact 0.2: Anaïs Nin died on January 14, 1977.
  Fact 0.3: Anaïs Nin was an essayist.
  Fact 0.4: Anaïs Nin was a memoirist.
  Fact 0.5: Anaïs Nin was born to Cuban parents.
  Fact 0.6: Anaïs Nin was born in France.
  Fact 0.7: Anaïs Nin was raised in France.
Sentence 1: Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as a major exponent of romantic modernism in Greece and the world.
  Fact 1.0: Odysseas Elytis is a pen name of Odysseas Alepoudellis.
  Fact 1.1: Odysseas Elytis's Greek name is Οδυσσέας Ελύτης.
  Fact 1.2: Odysseas Alepoudellis's Greek name is Οδυσσέας Αλεπουδέλλης.
  Fact 1.3: Odysseas Elytis was born on 2 November 1911.
  Fact 1.4: Odysseas Elytis died on 18 March 1996.
  Fact 1.5: Odysseas Elytis was regarded as a major exponent of romantic modernism in Greece.
  Fact 1.6: Odysseas Elytis was regarded as a major exponent of romantic modernism in the world.
Sentence 2:  In 1979 he was awarded the Nobel Prize in Literature.
  Fact 2.0: Odysseas Elytis was awarded the Nobel Prize in Literature in 1979.

Retrieved items:
<item 1>
Collages (novel)
Collages, published in 1964, was Anaïs Nin's last published novel (excluding her erotica).

 It is very different from the previous novels of the "Cities of the Interior" series, because it contains none of the familiar characters in those novels.

 It is also different in that it takes place on two continents, has about two dozen important characters, and for the first time in Anaïs Nin's work the female
lead is not seeking psychological wholeness.

 Most of Anaïs Nin's other novels only have a few main characters, there is rarely much geographical movement in the other novels, and usually the lead
character is a woman seeking psychological wholeness.
</item 1>

<item 2>
The Diary of Anaïs Nin
The Diary of Anaïs Nin is the published version of Anaïs Nin's own private manuscript diary, which she began at age 11 in 1914 during a trip from Europe to New
York with her mother and two brothers.

 Anaïs Nin would later say she had begun the diary as a letter to her father, Cuban composer Joaquín Nin, who had abandoned the family a few years earlier.
</item 2>

<item 3>
Anaïs Nin: A Biography
Anaïs Nin: A Biography is Deirdre Bair's award-winning biography of writer Anaïs Nin.

 It is considered arguably by many to be the most comprehensive, well-researched, and scholarly biography available of Nin.

 Though the biography has received praise, it has also angered some fans of Anaïs Nin as well as some of her former associates, some of whom claim that Bair's
critical and rigorous investigation of Anaïs Nin's life is unsympathetic.
</item 3>

<item 4>
Fire: From a Journal of Love
Fire: From the Unexpurgated Diary of Anaïs Nin (full title "Fire: From A Journal of Love: the Unexpurgated Diary of Anaïs Nin (1934–1937)") is a 1995 book that
is based on material excerpted from the unpublished diaries of Anais Nin.

 It corresponds temporally to part of Anaïs Nin's published diaries, but consists mostly of material about her love life that was too sensitive or secret to
publish in her lifetime or in that of others involved.
</item 4>

<item 5>
Incest: From a Journal of Love
Incest: From a Journal of Love: The Unexpurgated Diary of Anaïs Nin (1932–1934) is a 1992 non-fiction book by Anaïs Nin.

 It is a continuation of the diary entries first published in "Henry and June: From the Unexpurgated Diary of Anaïs Nin".

 It features Nin's relationships with writer Henry Miller, his wife June Miller, the psychoanalyst Otto Rank, her father Joaquín Nin, and her husband Hugh
Parker Guiler.

 She also copied some of her correspondence with these people into her diary.

 Much of this book was written in English, although those of her letters which were originally written in French and Spanish were translated.

 Most of this diary takes place in France, particularly Clichy, Paris and Louveciennes.
</item 5>

<item 6>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 6>

<item 7>
Odysseas Elytis
Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as
a major exponent of romantic modernism in Greece and the world.

 In 1979 he was awarded the Nobel Prize in Literature.
</item 7>

<item 8>
Seduction of the Minotaur
Seduction of the Minotaur is an autobiographical novel by the mixed nationality writer Anaïs Nin, the last part of her "Cities of the Interior" sequence.

 It is about a woman named Lillian, and her self-psychoanalysis.

 The setting is taken from Anaïs' diary account of her first trip to Acapulco in 1947, and the novel repeats much of the first part of "The Diary of Anaïs Nin"
volume V.

 Since the author was concerned with psychology rather than physical adventure, there is actually less violence in the novel than in the diary account.

 The exception is that the doctor allows himself to be shot because he is loved only as a doctor and never as a man, perhaps patterned after her understanding
of Otto Rank's death.
</item 8>

<item 9>
Nearer the Moon
Nearer the Moon: From a Journal of Love (full title "Nearer the Moon: From a Journal of Love, the Unexpurgated Diary of Anaïs Nin (1937–1939)") is a 1996 book
based upon material excerpted from the unpublished diaries of Anaïs Nin.

 It corresponds temporally to part of Nin's published diaries.

 It consists mainly of material that was left out of the published version because it would have hurt people involved or their relationships with Anaïs Nin had
it been published at the time.
</item 9>

<item 10>
Anaïs Nin
Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban
parents in France, where she was also raised.

 She spent some time in Spain and Cuba, but lived most of her life in the United States, where she became an established author.

 She wrote journals (which span more than 60 years, beginning when she was 11 years old and ending shortly before her death), novels, critical studies, essays,
short stories, and erotica.

 A great deal of her work, including "Delta of Venus" and "Little Birds", was published posthumously.
</item 10>
=== END ITEM id=28c8e8dc8d930bcb ===
