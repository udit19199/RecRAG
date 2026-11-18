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

=== ITEM id=2531885f7b4b30e6 ===
Gold sentences and facts:
Sentence 0:  He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.
  Fact 0.0: The writer has written three novels.
  Fact 0.1: The writer has written three collections of short stories.
  Fact 0.2: The writer has written a collection of novellas.
  Fact 0.3: The writer has written seven volumes of essays.
  Fact 0.4: Three of the writer's essay volumes have won National Book Critics Circle Award prizes.
  Fact 0.5: One of the writer's essay volumes, 'A Temple of Texts' (2006), won the Truman Capote Award for Literary Criticism.
Sentence 1: Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for Literature.
  Fact 1.0: Naguib Mahfouz was born on December 11, 1911.
  Fact 1.1: Naguib Mahfouz died on August 30, 2006.
  Fact 1.2: Naguib Mahfouz's Arabic name is نجيب محفوظ, transliterated Nagīb Maḥfūẓ.
  Fact 1.3: Naguib Mahfouz was an Egyptian writer.
  Fact 1.4: Naguib Mahfouz won the 1988 Nobel Prize for Literature.

Retrieved items:
<item 1>
Frances Liardet
Frances E. Liardet is a writer and translator of Arabic literature.

 She studied Creative Writing at the University of East Anglia.

 She has translated several book-length works, including two books by the modernist Egyptian writer Edwar al-Kharrat and one by Nobel Prize winner Naguib
Mahfouz.
</item 1>

<item 2>
The Beginning and the End (novel)
The Beginning and the End (بداية ونهاية) is a novel by Naguib Mahfouz, who was awarded the Nobel Prize in literature in 1988.

 An Egyptian, Naguib has been credited with modernizing Arabic literature, with his prolific writing style and his themes on existentialism.

 The novel is marked by very bold characterization by Naguib Mahfouz, for the time period that the story is set in.

 The story moves at a prolific pace and tries to look at the world from each characters view point.
</item 2>

<item 3>
Naguib Mahfouz Medal for Literature
The Naguib Mahfouz Medal for Literature is a literary award for Arabic literature.

 It is given to the best contemporary novel written in Arabic, but not available in English translation.

 The winning book is then translated into English, and published by American University in Cairo Press.

 It was first awarded in 1996 and is presented annually on December 11, the birthday of Nobel laureate Naguib Mahfouz, by the President of the American
University in Cairo.
</item 3>

<item 4>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 4>

<item 5>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 5>

<item 6>
Naguib Mahfouz
Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for
Literature.

 He is regarded as one of the first contemporary writers of Arabic literature, along with Tawfiq el-Hakim, to explore themes of existentialism.

 He published 34 novels, over 350 short stories, dozens of movie scripts, and five plays over a 70-year career.

 Many of his works have been made into Egyptian and foreign films.
</item 6>

<item 7>
The entries cover major Arab cultural figures and works: Egyptian actor Ahmed Zaki and director Salah Abu Seif, known for intense performances and Egyptian
neorealism; and writer Naguib Mahfouz, Nobel laureate whose novels include *The Cairo Trilogy*, *The Beginning and the End*, *Thebes at War* and *Arabian Nights
and Days*. Abu Seif’s 1960 *Bidaya wa Nihaya* adapted Mahfouz. Other topics include Tayeb
</item 7>

<item 8>
Cairo Trilogy
The Cairo Trilogy (Arabic: الثلاثية‎ ‎ ("The Trilogy") or ثلاثية القاهرة ("The Cairo Trilogy")) is a trilogy of novels written by the Egyptian novelist and
Nobel Prize winner Naguib Mahfouz, and one of the prime works of his literary career.
</item 8>

<item 9>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 9>

<item 10>
Thebes at War
Thebes at War is an early novel by the Egyptian writer Naguib Mahfouz.

 It was originally published in Arabic in 1944.

 An English translation by Humphrey Davies appeared in 2003.

 The novel is one of several that Mahfouz wrote at the beginning of his career, with Pharaonic Egypt as their setting.

 Others in this series of novels include "Khufu's Wisdom" (1939) and "Rhadopis of Nubia" (1943).

 All have been translated into English and appeared in one volume under the title "Three Novels of Ancient Egypt" (Everyman's Library, 2007).
</item 10>
=== END ITEM id=2531885f7b4b30e6 ===

=== ITEM id=e64fe8bc3560c2b9 ===
Gold sentences and facts:
Sentence 0: Bank Street () is a short street in Central, Hong Kong.
  Fact 0.0: Bank Street is a short street.
  Fact 0.1: Bank Street is located in Central, Hong Kong.
Sentence 1:  It links Des Voeux Road Central to Queen's Road Central.
  Fact 1.0: Bank Street links Des Voeux Road Central to Queen's Road Central.
Sentence 2:  It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong) respectively.
  Fact 2.0: Bank Street is named after its location between the HSBC Building and the Bank of China Building.
  Fact 2.1: The HSBC Building was the former headquarters of HSBC.
  Fact 2.2: The Bank of China Building was the former headquarters of Bank of China (Hong Kong).
Sentence 3: HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC Holdings.
  Fact 3.0: HSBC Main Building is a headquarters building of The Hongkong and Shanghai Banking Corporation.
  Fact 3.1: The Hongkong and Shanghai Banking Corporation is today a wholly owned subsidiary of London-based HSBC Holdings.

Retrieved items:
<item 1>
HSBC
HSBC Holdings PLC is a British multinational banking and financial services holding company, tracing its origin to a hong in Hong Kong.

 It is the world's seventh largest bank by total assets and the largest in Europe with total assets of US$2.374 trillion (as of December 2016).

 It was established in its present form in London in 1991 by The Hongkong and Shanghai Banking Corporation Limited to act as a new group holding company.

 The origins of the bank lie mainly in Hong Kong and to a lesser extent in Shanghai, where branches were first opened in 1865.

 The HSBC name is derived from the initials of the Hongkong and Shanghai Banking Corporation.

 The company was first formally incorporated in 1866.

 The company continues to see both the United Kingdom and Hong Kong as its "home markets".
</item 1>

<item 2>
Bank of Canton
Bank of Canton () was established in 1912 in Canton but registered as a British company in Hong Kong.

 This made it the first Chinese-owned bank in Hong Kong.

 At the time, all the other banks in Hong Kong were foreign, primarily British, including the locally incorporated but British-run Hongkong and Shanghai Banking
Corporation.

 Security Pacific National Bank (SPNB) bought a majority share in Bank of Canton in 1971.

 In 1988, Security Pacific succeeded in wholly acquiriing Bank of Canton, which became Security Pacific Asia Bank.

 When Bank of America acquired Security Pacific Corporation in 1993 it changed Security Pacific Asia Bank’s name to Bank of America (Asia).

 China Construction Bank acquired Bank of America (Asia) in 2006.
</item 2>

<item 3>
Bank Street (Hong Kong)
Bank Street () is a short street in Central, Hong Kong.

 It links Des Voeux Road Central to Queen's Road Central.

 It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong)
respectively.
</item 3>

<item 4>
HSBC lions
Various headquarters and branch buildings of The Hongkong and Shanghai Banking Corporation and the HSBC Group, into which the Hongkong and Shanghai Banking
Corporation has evolved, feature a pair of lion sculptures.

 The HSBC lions have become distinctive landmarks in their own right in Hong Kong and Shanghai respectively, with a further pair to be found in London.
</item 4>

<item 5>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 5>

<item 6>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 6>

<item 7>
HSBC Building, the Bund
The HSBC Building (also known as the Municipal Government Building) is a six-floor neo-classical building in the Bund area of Shanghai, China.

 It served as the headquarters of the Shanghai branch of The Hongkong and Shanghai Banking Corporation from 1923 to 1955, and currently houses the Shanghai
Pudong Development Bank.

 The building is situated at number 12, the Bund.

 Designed by the British architecture firm Palmer & Turner Architects and Surveyors, construction of the building lasted from 5 May 1921 to 23 June 1923.
</item 7>

<item 8>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 8>

<item 9>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 9>

<item 10>
HSBC Building (Hong Kong)
HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC
Holdings.

 It is located along the southern side of Statue Square near the location of the old City Hall, Hong Kong (built in 1869, demolished in 1933).

 The previous HSBC building was built in 1935 and pulled down to make way for the current building.

 The address remains as 1 Queen's Road Central (the north facing side of the building was served by Des Voeux Road, which was seashore, making Queen's Road was
the main entrance, in contrast to modern day that usually accessed the bank from Des Voeux Road).

 The building can be reached from Exit K of Central MTR Station and facing Statue Square.
</item 10>
=== END ITEM id=e64fe8bc3560c2b9 ===

=== ITEM id=7a36221c72172d58 ===
Gold sentences and facts:
Sentence 0:  He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.
  Fact 0.0: He won three singles titles.
  Fact 0.1: He won 26 doubles titles.
  Fact 0.2: He won the Australian Open 1978 doubles title with Wojtek Fibak.
  Fact 0.3: He won the Australian Open 1980 title with fellow countryman Mark Edmondson.
  Fact 0.4: He won the Australian Open 1981 title with fellow countryman Mark Edmondson.
  Fact 0.5: He won the Roland Garros 1986 title with fellow countryman Mark Edmondson.
  Fact 0.6: He was a runner-up at the Australian Open 1985 with fellow countryman Mark Edmondson.
  Fact 0.7: Mark Edmondson is a fellow countryman of his.
Sentence 1:  In his career, he won nine doubles titles, but no singles titles.
  Fact 1.0: In his career, he won nine doubles titles.
  Fact 1.1: In his career, he won no singles titles.

Retrieved items:
<item 1>
Jorge Lozano
In Davis Cup play, represented Mexico for 15 years, 1981–1995, won 12 doubles matches, and lost 12 as well.

 He lost 11 singles matches, and won 8, making his overall win / loss record at the Davis Cup 20–23.

 Lozano won his first title at Forest Hills in 1988, with his partner Todd Witsken, and won his last at Athens in 1993.

 He resides in Guadalajara, Mexico where he directs a tennis academy for kids and also coaches the men and women tennis team at the University Tec de Monterrey.

 He is currently the captain of the Mexican Davis Cup Team.
</item 1>

<item 2>
1988 Volvo International – Doubles
Jorge Lozano and Todd Witsken won in the final 6–3, 7–6 against Pieter Aldrich and Danie Visser.
</item 2>

<item 3>
1994 Athens International – Doubles
Horacio de la Peña and Jorge Lozano were the defending champions, but did not participate this year.
</item 3>

<item 4>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 4>

<item 5>
1990 Stockholm Open – Doubles
Jorge Lozano and Todd Witsken were the defending champions, but lost in the quarterfinals this year.
</item 5>

<item 6>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 6>

<item 7>
1989 Volvo International – Doubles
Jorge Lozano and Todd Witsken were the defending champions but did not compete that year.
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
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 9>

<item 10>
1990 Canadian Open – Men's Doubles
Kelly Evernden and Todd Witsken were the defending champions but they competed with different partners that year, Evernden with Nicolás Pereira and Witsken with
Jorge Lozano.
</item 10>
=== END ITEM id=7a36221c72172d58 ===

=== ITEM id=6267386ca7aab0e4 ===
Gold sentences and facts:
Sentence 0:  Her favorite events are 800 metres and 1500 metres.
  Fact 0.0: This person's favorite events are the 800 metres.
  Fact 0.1: This person's favorite events are the 1500 metres.
Sentence 1:  The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.
  Fact 1.0: The distance has been contested at the Summer Olympics since 1896.
  Fact 1.1: The distance has been contested at the World Championships in Athletics since 1983.

Retrieved items:
<item 1>
Sinimole Paulose
Sinimol Paulose (born 24 June 1983) is an Indian middle distance runner.

 Her favorite events are 800 metres and 1500 metres.

 In 2009, she received Arjuna Award for her contribution to the Indian athletics.

 Sinimol was born in Piravom in Ernakulam district of Kerala state.

 She represented India at Beijing Olympics.
</item 1>

<item 2>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 2>

<item 3>
1500 metres
The 1500 metres or 1,500-metre run (typically pronounced 'fifteen-hundred metres') is the foremost middle distance track event in athletics.

 The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.

 It is equivalent to 1.5 kilometers or approximately ⁄  miles.
</item 3>

<item 4>
John Lindsay (Paralympian)
John Lindsay, OAM (born 29 January 1970) is an Australian Paralympic athlete from Melbourne.

 He competed in the 1988 Seoul games in distances ranging from 100 m to 800 m, but did not win any medals.

 At the 1992 Barcelona Games, he won a gold medal in the Men's 200 m TW3 event, for which he received a Medal of the Order of Australia, a silver medal in the
Men's 100 m TW3 event and a bronze medal in the Men's 400 m TW3 event.

 That year, he had a Victorian Institute of Sport scholarship.
</item 4>

<item 5>
Ross MacKenzie (athlete)
Ross MacKenzie (born 18 July 1946) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1968 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Brian MacLaren, Bill Crothers, and Robert McLaren).

 MacKenzie also finished fifth in the 1967 Pan American Games 400 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Brian MacLaren and Bill Crothers.
</item 5>

<item 6>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 6>

<item 7>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 7>

<item 8>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 8>

<item 9>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 9>

<item 10>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 10>
=== END ITEM id=6267386ca7aab0e4 ===

=== ITEM id=0a1f077e97398960 ===
Gold sentences and facts:
Sentence 0: Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France (1933–1940 and 1950–1957), and the United States (1947–1950).
  Fact 0.0: Maximillian Oppenheimer was born on 6 May 1902.
  Fact 0.1: Maximillian Oppenheimer died on 26 March 1957.
  Fact 0.2: Maximillian Oppenheimer was known as Max Ophüls.
  Fact 0.3: Max Ophüls was a German-born film director.
  Fact 0.4: Max Ophüls worked in Germany from 1931 to 1933.
  Fact 0.5: Max Ophüls worked in France from 1933 to 1940.
  Fact 0.6: Max Ophüls worked in France from 1950 to 1957.
  Fact 0.7: Max Ophüls worked in the United States from 1947 to 1950.
Sentence 1: Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.
  Fact 1.0: Shirley Clarke was born on October 2, 1919.
  Fact 1.1: Shirley Clarke died on September 23, 1997.
  Fact 1.2: Shirley Clarke was an American experimental filmmaker.
  Fact 1.3: Shirley Clarke was an American independent filmmaker.

Retrieved items:
<item 1>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 1>

<item 2>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 2>

<item 3>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 3>

<item 4>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
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
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 6>

<item 7>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 7>

<item 8>
Max Ophüls
Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France
(1933–1940 and 1950–1957), and the United States (1947–1950).

 He made nearly 30 films, the latter ones being especially notable: "La Ronde" (1950), "Le Plaisir" (1952), "The Earrings of Madame de..." (1953) and "Lola
Montès" (1955).
</item 8>

<item 9>
The Earrings of Madame de…
The Earrings of Madame de… (French: Madame de… ] ) is a 1953 drama film directed by Max Ophüls, adapted from Louise Leveque de Vilmorin's period novel by
Ophüls, Marcel Archard and Annette Wadement.

 The film is considered a masterpiece of the 1950s French cinema.

 Andrew Sarris called it "the most perfect film ever made".

 Ophüls said the story's construction attracted him, stating "there is always the same axis around which the action continually turns like a carousel.

 A tiny, scarcely visible axis: a pair of earrings."
</item 9>

<item 10>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 10>
=== END ITEM id=0a1f077e97398960 ===

=== ITEM id=d0f319f2938ee3b7 ===
Gold sentences and facts:
Sentence 0: Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
  Fact 0.0: Hänt Extra is a Swedish language magazine.
  Fact 0.1: Hänt Extra is a weekly magazine.
  Fact 0.2: Hänt Extra is a celebrity and youth magazine.
  Fact 0.3: Hänt Extra is published by Aller Media.
  Fact 0.4: Hänt Extra is published in Stockholm, Sweden.
Sentence 1: Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
  Fact 1.0: Tea Moderna is a Macedonian woman's weekly magazine.
  Fact 1.1: Tea Moderna's name in Macedonian is Теа Модерна.
  Fact 1.2: Tea Moderna is a magazine for women.

Retrieved items:
<item 1>
Katarina Ivanovska
Katarina Ivanovska (Macedonian: Катарина Ивановска ; born 18 August 1988) is a Macedonian model and actress.

 She began her modeling career in 2004, appearing at Milan Fashion Week after winning the Look Models International model search in Macedonia.

 In December, 2004, she appeared in a pictorial for "Elle" magazine and has also appeared in "Citizen K", "Stiletto" and the Italian and Russian "Vogue".

 She has been featured on the covers of "Diva" and "Maxima" magazines and in advertisements for D&G in 2006.

 She is considered the most successful Macedonian model.

 In 2010, Ivanovska appeared in Serbian "Elle" magazine.

 In 2011 she signed a contract for advertising Victoria's Secret products.

 In 2011 she got her first acting job in the Macedonian World War II film, "The Third Half", landing the lead role of a young Jewish girl named Rebecca.
</item 1>

<item 2>
Nikola Mladenov
Nikola Mladenov (Macedonian: Никола Младенов ; 10 March 1964 – 26 March 2013) was a Macedonian journalist, actor, founder and managing editor of the weekly
newspaper "Fokus", widely regarded as one of the most reputable journalists in Macedonia, who along with his colleagues from the newspaper "Mlad borec" is
considered to have laid the basis for the pluralistic principles in the Macedonian journalism.

 The "Award for best research story of the year – Nikola Mladenov", which is awarded by the Macedonian Institute for Media, is named after him.
</item 2>

<item 3>
Makedonsko Sonce
Makedonsko Sonce (Macedonian: ; English: Macedonian Sun ) is a weekly magazine published in the Republic of Macedonia.

 The title means "Macedonian Sun", referring to the Vergina Sun, which is used in the logo of the magazine.

 It was established by Gjorgija (George) Atanasoski and the first edition was published on June 24, 1994.

 It describes its own political stance as affirming "Macedonian national values, not only on territory of Macedonia but also in the other parts of ethnic
Macedonia".
</item 3>

<item 4>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 4>

<item 5>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 5>

<item 6>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 6>

<item 7>
Tea Moderna (magazine)
Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
</item 7>

<item 8>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 8>

<item 9>
2015 Macedonian Football Supercup
The 2015 Macedonian Supercup was the 3rd Macedonian Football Supercup, an annual Macedonian football match played between the winners of the previous season's
First League and Macedonian Cup.

 The game will be played between Rabotnički, who beat Teteks to win the 2015 Macedonian Cup Final, and Vardar, champions of the 2014–15 First League.

 It was played at Philip II Arena, Skopje and was won by Vardar by 4–3 on penalties after the match was ended 1–1 after the regular and extra time.
</item 9>

<item 10>
Albanians in the Republic of Macedonia
Albanians are the largest ethnic minority in the Republic of Macedonia.

 Of the 2,022,547 citizens of Macedonia, 509,083, or 25.2%, are Albanian according to the latest national census in 2002.

 The Albanian minority lives mostly in the north-western part of the country.

 The largest Albanian communities are in the municipalities of Tetovo (70.3% of the total population), Gostivar (66.7%), Debar (58.1%), Struga (56.8%), Kičevo
(54.5%), Kumanovo (25.8%) and Skopje (20.4%).
</item 10>
=== END ITEM id=d0f319f2938ee3b7 ===

=== ITEM id=d7902825dc95f521 ===
Gold sentences and facts:
Sentence 0: Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.
  Fact 0.0: Margaret Landon was born September 7, 1903.
  Fact 0.1: Margaret Landon died December 4, 1993.
  Fact 0.2: Margaret Landon was an American writer.
  Fact 0.3: Margaret Landon is best remembered for 'Anna and the King of Siam'.
  Fact 0.4: 'Anna and the King of Siam' is Margaret Landon's best-selling novel, published in 1944.
  Fact 0.5: 'Anna and the King of Siam' is a novel about the life of Anna Leonowens.
  Fact 0.6: 'Anna and the King of Siam' eventually sold over a million copies.
  Fact 0.7: 'Anna and the King of Siam' was translated into more than twenty languages.
Sentence 1:  In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.
  Fact 1.0: In 1950, Margaret Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II.
  Fact 1.1: Richard Rodgers and Oscar Hammerstein II created the musical 'The King and I' from Margaret Landon's book.
Sentence 2:  A later work, "Never Dies the Dream", appeared in 1949.
  Fact 2.0: 'Never Dies the Dream' is a later work.
  Fact 2.1: 'Never Dies the Dream' appeared in 1949.
Sentence 3: Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.
  Fact 3.0: Dame Elizabeth Rosemond Taylor, DBE, was born February 27, 1932.
  Fact 3.1: Dame Elizabeth Rosemond Taylor, DBE, died March 23, 2011.
  Fact 3.2: Dame Elizabeth Rosemond Taylor was a British-American actress.
  Fact 3.3: Dame Elizabeth Rosemond Taylor was a businesswoman.
  Fact 3.4: Dame Elizabeth Rosemond Taylor was a humanitarian.
Sentence 4:  She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.
  Fact 4.0: Elizabeth Taylor began her career as a child actress in the early 1940s.
  Fact 4.1: Elizabeth Taylor was one of the most popular stars of classical Hollywood cinema in the 1950s.
Sentence 5:  She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.
  Fact 5.0: Elizabeth Taylor continued her career successfully into the 1960s.
  Fact 5.1: Elizabeth Taylor remained a well known public figure for the rest of her life.
Sentence 6:  In 1999, the American Film Institute named her the seventh-greatest female screen legend.
  Fact 6.0: In 1999, the American Film Institute named Elizabeth Taylor the seventh-greatest female screen legend.

Retrieved items:
<item 1>
Krupp Diamond
The Elizabeth Taylor Diamond, formerly known as the Krupp Diamond, is a 33.19 carat diamond that was bought by Richard Burton for his wife, Elizabeth Taylor in
1968.

 The Krupp diamond was one of a number of significant pieces of jewellery owned by Taylor, her collection also included the 68 carat Taylor–Burton Diamond,
which was bought by the couple in 1969.

 The Krupp diamond was sold by Taylor's estate in 2011 for $8.8 million.
</item 1>

<item 2>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 2>

<item 3>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 3>

<item 4>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 4>

<item 5>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 5>

<item 6>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 6>

<item 7>
Cate Blanchett (born 14 May 1969) is an acclaimed Australian actress and theatre director who has won two Academy Awards, three Golden Globes, three BAFTAs, six
AACTA Awards and three Screen Actors Guild Awards. She rose to international prominence as Elizabeth I in *Elizabeth* (1998), won a supporting-actress Oscar for
portraying Katharine Hepburn in *The Aviator* (2004), and earned a best-actress Oscar for *Blue
</item 7>

<item 8>
Anna and the King of Siam (novel)
Anna and the King is a 1944 semi-fictionalized biographical novel by Margaret Landon.
</item 8>

<item 9>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 9>

<item 10>
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 10>
=== END ITEM id=d7902825dc95f521 ===

=== ITEM id=8542001cd9f66981 ===
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
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 1>

<item 2>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 2>

<item 3>
Leeds band Vib Gyor took its name from rainbow colors ordered by wavelength. Its debut single “Fallen” (July 2006) gained acclaim and radio play. The Secret EP
(March 2007), recorded in Bath by Paul Corkett, includes a hidden instrumental; The White EP followed in November, recorded at Rockfield by Ken Thomas. Leeds
alternative-rock band Distortion Mirrors formed in 2011. Other releases include Edward Sharpe and the Magnetic Zeros’ 201
</item 3>

<item 4>
The White EP (Vib Gyor EP)
The "White EP" is a promo EP from the Leeds band Vib Gyor.

 It was first sold on their UK tour with American band Transfer and was released through their My Space page and official web-site on 13 November 2007.

 It is the follow-up to their The Secret EP.

 It was recorded in Rockfield studios in South Wales and produced by Ken Thomas.
</item 4>

<item 5>
Fallen (Vib Gyor song)
"Fallen" was the first single from the Leeds band Vib Gyor.

 It was first given limited release in July 2006, and has attained critical acclaim and a large amount of radio play on both sides of the atlantic.
</item 5>

<item 6>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 6>

<item 7>
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
</item 7>

<item 8>
The Gorgeous
The Gorgeous were a Canadian metalcore band formed in the summer of 2003 in Montreal.

 The band formed with Dali Shaw on guitar, Julien Brousseau on Bass and Miguel Shaw on drums.

 The three played a few shows without a vocalist and quickly recruited Jordon Daniel.
</item 8>

<item 9>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 9>

<item 10>
The Famines (band)
The Famines are a Canadian indie rock band formed in 2008 in Edmonton, Alberta, Canada now based in Montreal, Quebec, Canada.

 The two piece band uses a modern and minimalistic approach that draws comparison to mid 1970's protopunk and fuzzy garage rock.

The band name is meant to be a commentary on the continued feeling of emptiness and lacking in a society that is materially fulfilled.

 The band has two members, R. E. Biesinger on guitar and vocals, and Drew Demers on the drums.
</item 10>
=== END ITEM id=8542001cd9f66981 ===

=== ITEM id=94579f64b1c93ebc ===
Gold sentences and facts:
Sentence 0: Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.
  Fact 0.0: Triumph of the Will's German title is Triumph des Willens.
  Fact 0.1: Triumph of the Will is a 1935 German propaganda film.
  Fact 0.2: Leni Riefenstahl directed Triumph of the Will.
  Fact 0.3: Leni Riefenstahl produced Triumph of the Will.
  Fact 0.4: Leni Riefenstahl edited Triumph of the Will.
  Fact 0.5: Leni Riefenstahl co-wrote Triumph of the Will.
Sentence 1: The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".
  Fact 1.0: The End of America is a 2008 documentary film.
  Fact 1.1: Annie Sundberg directed The End of America.
  Fact 1.2: Ricki Stern directed The End of America.
  Fact 1.3: The End of America is adapted from Naomi Wolf's 2007 book.

Retrieved items:
<item 1>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 1>

<item 2>
The End of America (film)
The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".

 The film analyzes policy changes made during the Bush Administration, and makes the case that these changes threaten American democracy.
</item 2>

<item 3>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 3>

<item 4>
Empire (1964 film)
Empire is a 1964 American black and white silent film written, produced, and directed by Andy Warhol.

 It consists of eight hours and five minutes of continuous slow motion footage of the Empire State Building in New York City.

 Abridged showings of the film were never allowed, and supposedly the unwatchability of the film was an important part of the reason the film was created.

 However, a legitimate Italian VHS produced in association with the Andy Warhol Museum in 2000 contains only an extract of 60 minutes.

 Its use of the long take "in extremis" is an extension of Warhol's earlier work the previous year with "Sleep".

 Warhol employed Rob Trains to be the projectionist for a screening of the film.

 Trains miscalculated and mixed the order and speed of the reels for the eight-hour movie.

 After a positive review in "The New York Times", Warhol actually liked the "mistake" and employed Trains for the entire summer.
</item 4>

<item 5>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 5>

<item 6>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 6>

<item 7>
Kaiserjäger (film)
Kaiserjäger is a 1956 Austrian film directed by Willi Forst.
</item 7>

<item 8>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 8>

<item 9>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 9>

<item 10>
Triumph of the Will
Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.

 It chronicles the 1934 Nazi Party Congress in Nuremberg, which was attended by more than 700,000 Nazi supporters.

 The film contains excerpts from speeches given by Nazi leaders at the Congress, including Adolf Hitler, Rudolf Hess and Julius Streicher, interspersed with
footage of massed Sturmabteilung (SA) and Schutzstaffel (SS) troops and public reaction.

 Hitler commissioned the film and served as an unofficial executive producer; his name appears in the opening titles.

 The film's overriding theme is the return of Germany as a great power, with Hitler as the leader who will bring glory to the nation.

 Because the film was made after the 1934 Night of the Long Knives (on 30 June) many prominent Sturmabteilung (SA) members are absent—they were murdered in that
Party purge, organised and orchestrated by Hitler to replace the SA with the Schutzstaffel (SS) as his main paramilitary force.
</item 10>
=== END ITEM id=94579f64b1c93ebc ===

=== ITEM id=66e8dbcb1a1edf90 ===
Gold sentences and facts:
Sentence 0:  It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
  Fact 0.0: This song is most known through the 1989 single by Kathy Mattea.
  Fact 0.1: The 1989 single by Kathy Mattea was released in conjunction with her album Willow in the Wind.
  Fact 0.2: This song was first recorded and released on the 1987 Don Williams album Traces.
  Fact 0.3: This song was also released in 1988 by Clark's husband on his album Old Friends.
Sentence 1: This is a detailed discography for American country music artist Don Williams.
  Fact 1.0: The discography is detailed.
  Fact 1.1: The discography is for American country music artist Don Williams.

Retrieved items:
<item 1>
Come On (Earl King song)
"Come On" (often called "Let the Good Times Roll") is a song written by New Orleans rhythm and blues artist Earl King.

 He first recorded the song as "Darling Honey Angel Child" in 1960 for the Ace Records subsidiary Rex.

 Later that year, he recorded it as a two-part song for Imperial Records using some new lyrics.

 Retitled "Come On", it was released in 1960 with "Come On - Part I” as the A-side backed with “Come On - Part II” (Imperial 5713).
</item 1>

<item 2>
Hjärtats trakt
Hjärtats trakt was released on 20 April 1993 and is a compilation album from Swedish pop artist Per Gessle.

 The album includes songs taken from Gessle's solo albums "Per Gessle" and "Scener".

 It was also released under license by the British record company Pickwick and it is the only Gessle album released by another label than EMI.

 In 1998 the album was re-released by the Dutch record company Disky Communications B.V. which is a subsidiary of EMI.
</item 2>

<item 3>
Night Is Fallin' in My Heart
"Night Is Fallin' in My Heart" is a song written by Dennis Linde.

 The song was first recorded by Country music artist J.P. Pennington for his 1991 album, "Whatever It Takes".

 It was later recorded and released as a single in October 1994 by American country music group Diamond Rio.

 It was the second single released from their third album, "Love a Little Stronger".

 It peaked at number 9 in the United States, and number 6 in Canada.
</item 3>

<item 4>
Every Beat of My Heart (song)
"Every Beat of My Heart" is a rhythm and blues song by Johnny Otis.

 It was first recorded in 1954 by his group, The Royals (later to be known as The Midnighters).

 In 1961 Gladys Knight & the Pips recorded the song for their debut single on the Vee-Jay label.

 Credited to The Pips, it was the first of eleven releases by the group to make it to number one on the R&B/soul chart.

 It was also the group's first top ten on the "Billboard" Hot 100.

 "Every Beat of My Heart" was first recorded for the Huntom label, who later sold the master to Vee-Jay.

 At the time of the song's release, The Pips were on the Fury label where they re-recorded the song without piano.

 In an unusual occurrence, the Fury recording of the song also made the top twenty on the R&B Sides chart and also made the Hot 100.
</item 4>

<item 5>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 5>

<item 6>
Patti Page
Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country
music.

 She was the top-charting female vocalist and best-selling female artist of the 1950s, selling over 100 million records during a six-decade long career.

 She was often introduced as "the Singin' Rage, Miss Patti Page".

 New York WNEW disc-jockey William B. Williams introduced her as "A Page in my life called Patti".
</item 6>

<item 7>
Don Williams discography
This is a detailed discography for American country music artist Don Williams.
</item 7>

<item 8>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 8>

<item 9>
Progressive country
Progressive country is a subgenre of country music developed in the early 1970s.

 In the late 1960s and early 1970s, mainstream country music was dominated by the slick Nashville sound and the rock-influenced Bakersfield sound of artists
like Merle Haggard.

 A new generation of country artists emerged, influenced by contemporary rock music, singer-songwriters such as Bob Dylan, and the liberal politics of the 1960s
counterculture.
</item 9>

<item 10>
Refugees of the Heart
Refugees of the Heart is the sixth solo studio album by Steve Winwood, released in 1990.

 The album contained the hit single, "One and Only Man", which topped the Mainstream Rock Tracks chart and saw the return of former Traffic bandmate Jim Capaldi
to Winwood's songwriting team.

 By coincidence, a Traffic reunion followed in 1994.

 Because of that collaboration, Winwood would not record another solo album until late 1997.

 Also "I Will Be Here" and "Another Deal Goes Down" were released as singles.

 Winwood stated about the closing track, “In The Light Of Day”: 'When Will and I wrote the song (.

.)

 it was our idea of what Nelson Mandela's dream was, while he was in prison.

 It was really just a fantasy of ours, but that’s what we based the song on.'
</item 10>
=== END ITEM id=66e8dbcb1a1edf90 ===

=== ITEM id=71a0c405f5be1ed1 ===
Gold sentences and facts:
Sentence 0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie, Michigan.
  Fact 0.0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant.
  Fact 0.1: The Saint Marys Falls Hydropower Plant is a Cloverland Electric Cooperative hydroelectric plant.
  Fact 0.2: The Saint Marys Falls Hydropower Plant is in Sault Ste. Marie, Michigan.
Sentence 1:  Excavation of the power canal began in September 1898 and was completed in June 1902.
  Fact 1.0: Excavation of the Edison Sault Power Canal began in September 1898.
  Fact 1.1: Excavation of the Edison Sault Power Canal was completed in June 1902.
Sentence 2:  The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
  Fact 2.0: The canal was named a Historic Civil Engineering Landmark in 1983.
  Fact 2.1: The hydroelectric complex was named a Historic Civil Engineering Landmark in 1983.
Sentence 3: The Cayuga–Seneca Canal is a canal in New York, United States.
  Fact 3.0: The Cayuga–Seneca Canal is a canal in New York, United States.
Sentence 4:  It is now part of the New York State Canal System.
  Fact 4.0: The Cayuga–Seneca Canal is now part of the New York State Canal System.

Retrieved items:
<item 1>
The entries cover historic U.S. canals, waterways, water systems, and sites. Transport canals include Beaver–Erie (136 miles), Delaware (60), Whitewater (76),
and Susquehanna–Tidewater (43), plus interstate and regional routes; many declined with railroad competition. The Dismal Swamp Canal, opened in 1805 and closed
in 2016, served the Intracoastal Waterway; nearby park and refuge protect the swamp.
</item 1>

<item 2>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 2>

<item 3>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 3>

<item 4>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 4>

<item 5>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 5>

<item 6>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 6>

<item 7>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 7>

<item 8>
Erie Canal Lock 52 Complex
Erie Canal Lock 52 Complex is a national historic district located at Port Byron and Mentz in Cayuga County, New York.

 The district includes two contributing buildings (the Erie House and the blacksmith shop / mule barn); three contributing engineering structures (Erie Canal
Lock 52, culvert, and canal prism of the enlarged Erie Canal); and archaeological sites associated with the canal operations.

 Lock 52 was constructed 1849-1853 as part of the Enlarged Erie Canal program.

 It remained in operation until the rerouting of the canal under the New York State Barge Canal System in 1917.

 The Erie House was built in 1894 and is a two story frame structure that housed a saloon and hotel.
</item 8>

<item 9>
New York’s Erie Canal heritage includes Lock 52 (1849–53), canal buildings and archaeological sites; the 1836–42 Broad Street Aqueduct later became a road and
subway route. The 1905 song “Low Bridge” recalls mule-powered canal traffic. The 525-mile state canal system includes the Erie, Oswego, Cayuga–Seneca and
Champlain canals. The Genesee River and tributaries Black Creek and Oatka Creek drain western New
</item 9>

<item 10>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 10>
=== END ITEM id=71a0c405f5be1ed1 ===

=== ITEM id=882513bddb073bfc ===
Gold sentences and facts:
Sentence 0:  For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record when her album "25" took just 29 days to reach the landmark.
  Fact 0.0: For almost 20 years, a record (held by a subject not named in this sentence) stood for selling 2 million copies in the UK in the fastest time.
  Fact 0.1: The record for selling 2 million copies in the UK in the fastest time was set in 42 days.
  Fact 0.2: In 2015, Adele surpassed the record.
  Fact 0.3: Adele's album '25' took just 29 days to reach the 2 million copies landmark.
Sentence 1:  At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
  Fact 1.0: The 51st Grammy Awards were held in 2009.
  Fact 1.1: Adele received the Best New Artist award at the 51st Grammy Awards.
  Fact 1.2: Adele received the Best Female Pop Vocal Performance award at the 51st Grammy Awards.

Retrieved items:
<item 1>
Billboard’s 2002 chart ranked U.S. albums using Nielsen SoundScan weekly sales. Twenty-five acts reached No. 1. Eminem’s *The Eminem Show* led the year with
about 7.6 million sales, six nonconsecutive weeks at No. 1, and a 1.322-million opening week. Other notable debuts included Shania Twain’s *Up!* (857,000),
Nelly’s *Nellyville* (714,000
</item 1>

<item 2>
Billboard Year-End
The same method was used for albums based on the "Billboard" 200, and songs appearing on the other charts (e.g. Hot Country Singles).
</item 2>

<item 3>
List of Billboard 200 number-one albums of 2002
The highest-selling albums and EPs in the United States are ranked in the "Billboard" 200, published by "Billboard" magazine.

 The data are compiled by Nielsen Soundscan based on each album's weekly physical and digital sales.

 25 acts achieved number one albums during this year with artist such as Nelly and Shania Twain who had their albums debut at number one on the chart.

 Rapper Eminem's "The Eminem Show" is the best selling album of 2002 selling over approximately 7.6 million copies by the end of the year.

 It is also the longest running album of 2002 spending six non-consecutive weeks the chart and was known for its first full week of sales debut of 1.322 million
copies which Nielsen SoundScan scanned as the sixth largest sales of all time in its first week.

 Its debut of 1.322 million copies has still not been matched by any album today since except for Taylor Swift's album "1989", which opened with first week
sales of 1.279 million copies.
</item 3>

<item 4>
MTV Video Music Award for Best New Artist
The MTV Video Music Award for Best New Artist have been given out since the first annual MTV Video Music Awards in 1984.

 From then to 2006, the award was named Best New Artist in a Video.

 In 2007 its name was changed to Best New Artist, as the category underwent a format change to award the artist's body of work for the full year rather than a
specific video.

 For the 2008 ceremony, though, while the award retained its 2007 name, it returned to the format of awarding a specific video rather than the artist's full
body of work.

 The category was later renamed Artist to Watch through 2013 to 2015 while still keeping the format of an award going to a certain video.
</item 4>

<item 5>
Robson &amp; Jerome (album)
Robson & Jerome is the debut studio album by British singing duo Robson & Jerome.

 It was released in the UK in 1995 by RCA Records, and reached number one on the UK Albums Chart.

 It was the Christmas number one album of 1995, and was the biggest-selling album of the year, with sales of 2,040,000.

 For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record
when her album "25" took just 29 days to reach the landmark.
</item 5>

<item 6>
The Foundation (2008), Zac Brown Band’s major-label debut, sold 3.4 million US copies by 2015, earned Grammy and ACM nominations, and won the band a Best New
Artist Grammy. Tevin Campbell’s Babyface-produced “Can We Talk” reached Hot 100 #9, topped the R&B chart for three weeks, went gold, and won a Soul Train award.
John Cena’s 2005 album introduced his entrance theme; Bow Wow’s “Fresh Azim
</item 6>

<item 7>
Robson & Jerome’s 1995 debut topped the UK chart, selling 2.04 million copies; its 42-day record for reaching two million lasted until Adele’s *25*. Forbes’
Beatles tribute won Sweden’s 1977 selection but finished last at Eurovision. Dylan’s 1978 “Baby, Stop Crying” reached UK #13. Jobim’s “Corcovado” became “Quiet
Nights of Quiet Stars”; Andy Williams and Kitty Kallen recorded it. “
</item 7>

<item 8>
List of Billboard 200 number-one albums of 2002
Country music singer Shania Twain's album "Up!

" opened up with a huge first week sales of 857,000 copies in its first week alone, giving her the recognition of the highest first week sales of her career and
second highest of the year, only behind Eminem's "The Eminem Show" and at the time the fastest selling solo female album ever.

 Nelly's album "Nellyville" opened up with his highest first week sales of his career which logged on with huge sales of 714,000 copies in its first week alone,
which beat his sales of his debut album "Country Grammar", which opened up with first week sales of 235,000 copies.

 Country singer Alan Jackson album "Drive" gave him his first number one album on the chart and opened up with first week sales of 211,000 copies alone.
</item 8>

<item 9>
List of Billboard 200 number-one albums of 2002
The band Creed continued its eight week long run on the chart but is credited as the longest running album 2001.

 Jennifer Lopez earned her second number one album on the charts with "", which became the highest first week sales of a remix album at the time.

 R&B artist Ashanti earned her first number one album with her self-titled debut album "Ashanti", which opened up with first week sales of 503,000 copies in its
first week alone.

 Puff Daddy earned his first number one album since "No Way Out" back in 1997.

 Rapper Jay-Z earned his fifth chart topper with "", which opened up with first week sales of 545,000 copies alone.

 Heavy metal band Disturbed earned its first number one album on the chart with "Believe", which opened up with first week sales of 284,000 copies alone.

 Country music singer Shania Twain's album "Up!

"
</item 9>

<item 10>
Adele
Adele Laurie Blue Adkins (born 5 May 1988) ( ) is an English singer-songwriter.

 After graduating from the BRIT School for Performing Arts and Technology in 2006, Adele was given a recording contract by XL Recordings after a friend posted
her demo on Myspace the same year.

 In 2007, she received the Brit Awards "Critics' Choice" award and won the BBC Sound of 2008 poll.

 Her debut album, "19", was released in 2008 to commercial and critical success.

 It is certified seven times platinum in the UK, and three times platinum in the US.

 The album contains her first song, "Hometown Glory", written when she was 16, which is based on her home suburb of West Norwood in London.

 An appearance she made on "Saturday Night Live" in late 2008 boosted her career in the US.

 At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
</item 10>
=== END ITEM id=882513bddb073bfc ===
