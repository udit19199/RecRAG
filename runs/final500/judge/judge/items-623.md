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

=== ITEM id=e5261c5e7ef702c8 ===
Gold sentences and facts:
Sentence 0:  His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".
  Fact 0.0: The unnamed subject's American superior was General Dwight D. Eisenhower.
  Fact 0.1: Dwight D. Eisenhower wrote that the unnamed subject was 'blunt, at times to the point of rudeness'.
Sentence 1: Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United States from 1953 until 1961.
  Fact 1.0: Dwight David 'Ike' Eisenhower was born on October 14, 1890.
  Fact 1.1: Dwight David 'Ike' Eisenhower died on March 28, 1969.
  Fact 1.2: Dwight David 'Ike' Eisenhower was an American politician.
  Fact 1.3: Dwight David 'Ike' Eisenhower was an Army general.
  Fact 1.4: Dwight David 'Ike' Eisenhower served as the 34th President of the United States.
  Fact 1.5: Dwight David 'Ike' Eisenhower served as President from 1953 until 1961.

Retrieved items:
<item 1>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 1>

<item 2>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 2>

<item 3>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 3>

<item 4>
Presidency of Ronald Reagan
The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.

 Reagan, a Republican, took office as the 40th United States president following a landslide win over Democratic incumbent President Jimmy Carter in the 1980
presidential election.

 Reagan was succeeded by his vice president, George H. W. Bush, who won the 1988 presidential election with Reagan's support.
</item 4>

<item 5>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 5>

<item 6>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 6>

<item 7>
Bill Clinton
William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States
from 1993 to 2001.

 Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.

 Before that, he served as Arkansas attorney general, from 1977 to 1979.

 A member of the Democratic Party, Clinton was ideologically a New Democrat and many of his policies reflected a centrist "Third Way" political philosophy.
</item 7>

<item 8>
Dwight D. Eisenhower
Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United
States from 1953 until 1961.

 During World War II, he was a five-star general in the United States Army and served as Supreme Commander of the Allied Expeditionary Forces in Europe.

 He was responsible for planning and supervising the invasion of North Africa in Operation Torch in 1942–43 and the successful invasion of France and Germany in
1944–45 from the Western Front.
</item 8>

<item 9>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 9>

<item 10>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 10>
=== END ITEM id=e5261c5e7ef702c8 ===

=== ITEM id=ee56541339bc73f0 ===
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
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
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
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 4>

<item 5>
2013 Aegon Classic – Doubles
Tímea Babos and Hsieh Su-wei were the defending champions, but Babos decided not to participate.

<br>
</item 5>

<item 6>
Sharon Walraven
Sharon Walraven (born 19 June 1970, Schaesberg) is a Dutch wheelchair tennis player.

 She became paraplegic at age 23 after complications following a fall while she was ice-skating.

 She has won seven Grand Slams doubles titles partnering compatriot Esther Vergeer.

 At the 2008 Paralympics in Beijing she won the Gold medal in the women's doubles competition.

 At the 2000 Paralympics in Sydney she won a Silver medal in the women's singles competition.

 Walraven has a highest ranking of No.2 in singles and No.1 in doubles.
</item 6>

<item 7>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 7>

<item 8>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 8>

<item 9>
1974 Stadthalle Open – Singles
Vitas Gerulaitis won in the final 6–4, 3–6, 6–3, 6–2 against Andrew Pattison.
</item 9>

<item 10>
1979 World Championship Tennis Finals – Singles
Vitas Gerulaitis was the defending champion but lost in the semifinals to Björn Borg.
</item 10>
=== END ITEM id=ee56541339bc73f0 ===

=== ITEM id=fc4a7fcb93e2fd0e ===
Gold sentences and facts:
Sentence 0: Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.
  Fact 0.0: Guitar World is a monthly music magazine.
  Fact 0.1: Guitar World is devoted to guitarists.
  Fact 0.2: Guitar World has been published since July 1980.
Sentence 1: Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.
  Fact 1.0: Golf Digest is a monthly golf magazine.
  Fact 1.1: Golf Digest is published by Condé Nast Publications.
  Fact 1.2: Golf Digest is published in the United States.
Sentence 2:  The magazine started in 1950, and was sold to The New York Times Company in 1969.
  Fact 2.0: The magazine started in 1950.
  Fact 2.1: The magazine was sold to The New York Times Company in 1969.
Sentence 3:  The headquarters of "Golf Digest" is in Des Moines, Iowa.
  Fact 3.0: The headquarters of Golf Digest is in Des Moines, Iowa.

Retrieved items:
<item 1>
Graph query row:
publication: Guitar World
start_publication_date: July 1980
</item 1>

<item 2>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 2>

<item 3>
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
</item 3>

<item 4>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 4>

<item 5>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 5>

<item 6>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 6>

<item 7>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 7>

<item 8>
Graph query row:
publication: Golf Digest
start_publication_date: 1950
</item 8>

<item 9>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 9>

<item 10>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 10>
=== END ITEM id=fc4a7fcb93e2fd0e ===

=== ITEM id=ee00fe95ee45bfa3 ===
Gold sentences and facts:
Sentence 0: And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
  Fact 0.0: And There's More is a comedy sketch show.
  Fact 0.1: And There's More stars Jimmy Cricket.
  Fact 0.2: And There's More was produced by Central Independent Television.
  Fact 0.3: And There's More was produced for ITV.
  Fact 0.4: And There's More was produced from 1985 till 1988.
Sentence 1:  He currently lives with his family in Rochdale, North West England.
  Fact 1.0: The person described in this passage currently lives with his family in Rochdale, North West England.
  Fact 1.1: The person described in this passage currently lives in Rochdale.
  Fact 1.2: Rochdale is in North West England.

Retrieved items:
<item 1>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 1>

<item 2>
Catsmeat Potter-Pirbright
Claude Cattermole "Catsmeat" Potter-Pirbright is a recurring fictional character from the Jeeves stories of British comic writer P. G. Wodehouse, being a member
of the Drones Club and a longtime school friend of Jeeves's master Bertie Wooster.

 An actor also known as Claude Cattermole on stage or Claude Pirbright from his birth name Claude Cattermole Pirbright, he has the nickname Catsmeat (i.e.
lights - lung-based cat food).
</item 2>

<item 3>
List of Jimmy Fallon games and sketches
The following is a list of recurring games, sketches, and other comedy routines from the NBC late night talk show "The Tonight Show Starring Jimmy Fallon", and
its predecessor, "Late Night with Jimmy Fallon".

 The sketches feature host Jimmy Fallon, house band The Roots, announcer/sidekick Steve Higgins, the show's writers, celebrity guests, and audience members.

 Most of the skits below appeared only on "Late Night", while some have carried over to the "Tonight Show".
</item 3>

<item 4>
Mazzy Star
Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and
bassist Kendra Smith.

 Roback's friend Hope Sandoval became the group's vocalist when Smith left the band.
</item 4>

<item 5>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 5>

<item 6>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 6>

<item 7>
Jimmy Cricket
James Mulgrew (born 17 October 1945), known professionally as Jimmy Cricket, is an Irish comedian.

 He currently lives with his family in Rochdale, North West England.
</item 7>

<item 8>
Live in Old Smokey
Live in Old Smokey is a live CD by English singer/songwriter Linda Lewis recorded at Ronnie Scott's in London and issued in 2006.
</item 8>

<item 9>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 9>

<item 10>
A Cream Cracker under the Settee
"A Cream Cracker Under The Settee" is a dramatic monologue written by Alan Bennett in 1987 for television, as part of his "Talking Heads" series for the BBC.

 The series became very popular, moving onto BBC Radio, international theatre, becoming one of the best-selling audio book releases of all time and included as
part of both the A-level and GCSE English syllabus.

 It was the sixth and final episode of the first series of "Talking Heads".
</item 10>
=== END ITEM id=ee00fe95ee45bfa3 ===

=== ITEM id=19ae1e8443b1f301 ===
Gold sentences and facts:
Sentence 0: Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.
  Fact 0.0: Jelena Janković was born on 28 February 1985.
  Fact 0.1: Jelena Janković's name is written in Serbian Cyrillic as Јелена Јанковић.
  Fact 0.2: Jelena Janković is a Serbian professional tennis player.
Sentence 1: Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian politician and retired professional tennis player.
  Fact 1.0: Marat Mubinovich Safin's name is written in Russian as Мара́т Муби́нович Са́фин.
  Fact 1.1: Marat Mubinovich Safin's name is written in Tatar as Марат Мөбин улы Сафин.
  Fact 1.2: Marat Mubinovich Safin's Tatar name is romanized as Marat Möbin uğlı Safin.
  Fact 1.3: Marat Mubinovich Safin was born on 27 January 1980.
  Fact 1.4: Marat Mubinovich Safin is a Russian politician.
  Fact 1.5: Marat Mubinovich Safin is a retired professional tennis player.

Retrieved items:
<item 1>
Dinara Safina career statistics
This is a list of the main career statistics of retired, Russian professional tennis player, Dinara Safina.

 Throughout her career, Safina won twelve WTA singles titles including three Tier I singles titles at the 2008 Qatar Telecom German Open, Rogers Cup and Toray
Pan Pacific Open respectively; one Premier Mandatory singles title at the 2009 Mutua Madrileña Madrid Open and one Premier 5 singles title at the 2009
Internazionali BNL d'Italia.

 She was also the runner-up at the 2008 French Open and the 2009 Australian Open and French Open as well as a silver medalist in singles at the 2008 Beijing
Olympics.
</item 1>

<item 2>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 2>

<item 3>
Jelenin svet
Jelenin svet ("Jelena's World") is a 2008 independent documentary film written and directed by Tanja Brzaković, about former World No. 1 female tennis player,
Jelena Janković.
</item 3>

<item 4>
Ana Ivanovic career statistics
This is a list of the main career statistics of Serbian professional tennis player, Ana Ivanovic.

 Ivanovic won fifteen WTA singles titles including one grand slam singles title at the 2008 French Open and three WTA Tier I singles titles.

 She was also the runner-up at the 2007 French Open and 2008 Australian Open and a semi-finalist at the 2007 Wimbledon Championships and 2007 WTA Tour
Championships.

 On June 9, 2008 Ivanovic became the world No. 1 for the first time in her career.
</item 4>

<item 5>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 5>

<item 6>
2008 Wimbledon Championships – Mixed Doubles
Jamie Murray and Jelena Janković were the defending champions, but Janković chose not to participate, and only Murray competed that year.
</item 6>

<item 7>
Jelena Janković
Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.

 Janković is a former World No. 1 in singles, a ranking achieved preceding her finals appearance at the 2008 US Open.

 Janković's career highlights include winning the 2007 Wimbledon Mixed Doubles title with Jamie Murray, the 2010 Indian Wells Masters, the Internazionali BNL
d'Italia twice, in 2007 and 2008, and the aforementioned appearance in the finals of the 2008 US Open.

 She is coached by her brother Marko.
</item 7>

<item 8>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 8>

<item 9>
Ivanovic–Janković rivalry
Serbian tennis players Ana Ivanovic and Jelena Janković are both former world No. 1 and have met 12 times so far.

 Their overall head-to-head series is 9–3, in Ivanovic's favor.

 Janković is two years older than Ivanovic, so their paths connect them to play each other.

 They are the first female players to achieve major successes for independent Serbia.
</item 9>

<item 10>
Jelena Janković career statistics
This is a list of the main career statistics of Serbian professional tennis player and former World No. 1, Jelena Janković.

 To date, Janković has won 15 WTA singles titles including four WTA Tier I singles titles, one WTA Premier Mandatory singles title and one WTA Premier 5 singles
title.

 Janković was also the runner-up in singles at the 2008 US Open, a semi-finalist at the year-ending WTA Tour Championships in 2008, 2009 and 2013 and a
quarterfinalist at the 2008 Beijing Olympics.
</item 10>
=== END ITEM id=19ae1e8443b1f301 ===

=== ITEM id=5e994651e986fbbf ===
Gold sentences and facts:
Sentence 0: Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from 1997-1998.
  Fact 0.0: Robert Suettinger was the national intelligence officer for East Asia at the National Intelligence Council (NIC).
  Fact 0.1: Robert Suettinger held the role of national intelligence officer for East Asia for United States President Bill Clinton.
  Fact 0.2: Robert Suettinger held the role from 1997 to 1998.
Sentence 1: William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States from 1993 to 2001.
  Fact 1.0: William Jefferson Clinton was born William Jefferson Blythe III.
  Fact 1.1: William Jefferson Clinton was born on August 19, 1946.
  Fact 1.2: William Jefferson Clinton is an American politician.
  Fact 1.3: William Jefferson Clinton served as the 42nd President of the United States.
  Fact 1.4: William Jefferson Clinton served as President from 1993 to 2001.
Sentence 2:  Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.
  Fact 2.0: Before the presidency, William Jefferson Clinton was Governor of Arkansas.
  Fact 2.1: William Jefferson Clinton was Governor of Arkansas during two separate terms.
  Fact 2.2: William Jefferson Clinton's first term as Governor of Arkansas was from 1979 to 1981.
  Fact 2.3: William Jefferson Clinton's second term as Governor of Arkansas was from 1983 to 1992.

Retrieved items:
<item 1>
Susan Combs
Susan Combs (born February 26, 1945) is a Republican politician from the U.S. state of Texas, who served from 2007 to 2015 as the state's Comptroller of Public
Accounts.

 Prior to her tenure as Comptroller, Combs had served two terms as Commissioner of the Texas Department of Agriculture from 1999–2007, taking the reins as the
first woman elected to that office in 1998.

 Combs also served two terms in the Texas House of Representatives.

 On July 10, 2017, Combs was nominated by U.S. President Donald Trump to be the Assistant Secretary of the Interior for Policy, Management and Budget.
</item 1>

<item 2>
Wayne W. Lambert
The Brigadier General retired from the US Air Force in 1989.

 Lambert also holds an MBA from Chapman College (’76).

 In retirement, Lambert publicly supported U.S. presidential candidate Mitt Romney and U.S. vice presidential candidate Paul Ryan in the 2012 general election.
</item 2>

<item 3>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 3>

<item 4>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 4>

<item 5>
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
</item 5>

<item 6>
Lawrence Eagleburger
Lawrence Sidney Eagleburger (August 1, 1930 – June 4, 2011) was an American statesman and career diplomat, who served briefly as the Secretary of State under
President George H. W. Bush.

 Previously, he had served in lesser capacities under Presidents Richard Nixon, Jimmy Carter, and Ronald Reagan, and as Deputy Secretary of State under George
H. W. Bush.
</item 6>

<item 7>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 7>

<item 8>
Bill Clinton
William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States
from 1993 to 2001.

 Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.

 Before that, he served as Arkansas attorney general, from 1977 to 1979.

 A member of the Democratic Party, Clinton was ideologically a New Democrat and many of his policies reflected a centrist "Third Way" political philosophy.
</item 8>

<item 9>
George G. Finch
Army in September, 1941, with Major Finch as commander.

 After World War II, he was a leading critic of efforts to eliminate the air arm of the National Guard during peacetime.

 General Finch gained the respect and admiration of Air National Guardsmen throughout the nation with his steadfast support and successful efforts to preserve
the Air Guard.

 He became the first Chief of the Air Force Division of the National Guard Bureau in 1948.

 Under his leadership, the Air National Guard built to combat readiness and was among the first components called into service after the outbreak of the Korean
War.

 As a result of General Finch's vision and perseverance, 45,000 highly trained officers and airmen of 22 wings and 65 squadrons gave the Air Force the strength
it needed in the early, critical phases of the Communist drive down the Korean peninsula.General Finch served as the senior Air Force member of the United
Nations negotiating team at the peace talks at Panmunjom, Korea, and received the Legion of Merit for outstanding service in 1955; General Finch assumed command
of Fourteenth Air Force, Robins AFB, Georgia, becoming the nation's first Air National Guardsman to head a numbered air force.
</item 9>

<item 10>
Director of the Defense Intelligence Agency
The Director of the Defense Intelligence Agency is an intelligence officer who, upon nomination by the President and confirmation by the Senate, serves as the
nation's highest-ranking military intelligence officer.

 As the chief of the Defense Intelligence Agency, the Director is the primary intelligence adviser to the Secretary of Defense and the Chairman of the Joint
Chiefs of Staff and also answers to the Director of National Intelligence through the civilian Under Secretary of Defense for Intelligence.

 The Director is also the Commander of the Joint Functional Component Command for Intelligence, Surveillance and Reconnaissance, a subordinate command of United
States Strategic Command.

 Additionally, he chairs the Military Intelligence Board, which coordinates activities of the entire defense intelligence community.
</item 10>
=== END ITEM id=5e994651e986fbbf ===

=== ITEM id=e1bf1e1082dd8e90 ===
Gold sentences and facts:
Sentence 0: The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.
  Fact 0.0: The Night of the Arts is an annual event.
  Fact 0.1: The Night of the Arts is known in Finnish as 'Taiteiden yö'.
  Fact 0.2: The Night of the Arts is known in Swedish as 'Konstens natt'.
  Fact 0.3: The Night of the Arts is held in several major cities in Finland.
  Fact 0.4: The Night of the Arts is held in late August.
Sentence 1:  It is one of many White Night festivals held worldwide.
  Fact 1.0: The Night of the Arts is one of many White Night festivals.
  Fact 1.1: White Night festivals are held worldwide.
Sentence 2: The White Nights are a kind of all-night arts festival held in many cities in the summer.
  Fact 2.0: The White Nights are a kind of all-night arts festival.
  Fact 2.1: The White Nights festival is held in many cities.
  Fact 2.2: The White Nights festival is held in the summer.
Sentence 3:  The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are early and darkness is never complete.
  Fact 3.0: 'White nights' is the name given in areas of high latitude to the weeks around the summer solstice in June.
  Fact 3.1: During white nights, sunsets are late.
  Fact 3.2: During white nights, sunrises are early.
  Fact 3.3: During white nights, darkness is never complete.

Retrieved items:
<item 1>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 1>

<item 2>
- **Woody Guthrie Folk Festival:** Mid-July in Okemah, Oklahoma, honors Guthrie’s birth and legacy; organized by a nonprofit coalition, with indoor and outdoor
performances.
- **FLIP Animation Festival:** Held each November in Wolverhampton, UK; international screenings, talks, workshops and industry events. Founded in 2004; won a
2009 tourism award.
- **Green Man:** Independent four-day music and arts festival in Wales’ Brecon Beacons since
</item 2>

<item 3>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 3>

<item 4>
Night of the Arts
The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.

 It is one of many White Night festivals held worldwide.
</item 4>

<item 5>
Green Man Festival
Green Man is an independent music and arts festival held annually in mid-August in the Brecon Beacons, Wales since 2003.

 It has evolved into a 20,000 capacity 4-day event, showcasing predominantly live music (in particular alternative, indie, rock, folk, dance and americana),
with additional events showcasing literature, film, comedy, theatre and poetry.

 2015 saw 1,500 multi-arts acts perform across 17 stages.

 The festival site is divided into 10 areas, each offering a unique festival experience.

 Ceilidhs, all-night bonfires and secret gigs all add to the festival's unique identity.
</item 5>

<item 6>
Great Northern Arts Festival
The Great Northern Arts Festival is held each year in Inuvik, Northwest Territories.

 Lasting approximately ten days, it is an annual event held each summer.

 It was established in 1989 by Sharlene Alexander and Sue Rose.

 While the festival's mandate is to serve Northern artists and craftspeople, artists from around the world also attend.

 The festival is designed as a venue for artists to see each other's work and to experiment with new applications.

 It includes arts of all types, such as fiber art and Music of the Northwest Territories.

 Artists partake in demonstrations, masterclasses, seminars, and workshops.

 The festival's gallery offers pieces for sale.

 There are cultural presentations each evening which include concerts, dance, fashion, music, and storytelling.

 Media coverage brings festival highlights to viewing audiences.
</item 6>

<item 7>
White Night festivals
The White Nights are a kind of all-night arts festival held in many cities in the summer.

 The original festival is the White Nights Festival held in Saint Petersburg, Russia.

 The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are
early and darkness is never complete.

 In Saint Petersburg, the Sun does not set until after 10 p.m., and the twilight lasts almost all night.
</item 7>

<item 8>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 8>

<item 9>
Hellfest was a socially conscious U.S. hardcore festival known for barrier-free stage diving; canceled in 2005, it sparked refund lawsuits. CZW, founded in
1999, became a leading American ultraviolent wrestling promotion. Essence Festival, launched in 1995, celebrates Black music and culture, mainly in New Orleans.
Australia’s Easter Offshore Festival ran near Torquay through 2001. Newfoundland’s Atlantis Prize honors album artistry. Saint Nicholas Day gift-giving
</item 9>

<item 10>
Kerala School Kalolsavam
School Arts Festival of Kerala (Malayalam: കേരള സ്കൂൾ കലോത്സവം) is an annual event conducted by the state government of Kerala, featuring several art
competitions for high school and higher secondary school students of Kerala.

 The festival was started in 1956, and till 2008, it was called as "Kerala State School Youth Festival".

 The participants are students from classes 8th to 12th.

 Winners from different revenue districts for a particular event will be competing in state level competition.

 The event is usually conducted in December–January months of a year and is considered to be the biggest cultural event of Asia.

 57th edition of Kerala School Kalolsavam 2017 is scheduled for January 16 to 22 at Kannur.
</item 10>
=== END ITEM id=e1bf1e1082dd8e90 ===

=== ITEM id=bdbcad582c7e691e ===
Gold sentences and facts:
Sentence 0:  He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.
  Fact 0.0: He played college football at Hampton University.
  Fact 0.1: He attended Riverdale High School.
  Fact 0.2: Riverdale High School is in Riverdale, Georgia.
Sentence 1: Riverdale is a city in Clayton County, Georgia, United States.
  Fact 1.0: Riverdale is a city.
  Fact 1.1: Riverdale is in Clayton County, Georgia.
  Fact 1.2: Clayton County, Georgia is in the United States.

Retrieved items:
<item 1>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 1>

<item 2>
Two Rivers High School (Arkansas)
Two Rivers High School is a public high school for students in grades 7 through 12 located in Ola, Arkansas, United States.

 Two Rivers High School is administered by the Two Rivers School District.

 The Two Rivers High School and school district serves 650 sqmi of rural communities including Ola, Plainview, Rover, and Casa.

 Two Rivers High School is one of four high schools within Yell County and the sole high school administered by the Two Rivers School District.
</item 2>

<item 3>
Riverdale, Georgia
Riverdale is a city in Clayton County, Georgia, United States.

 The population was 15,134 at the 2010 census, up from 12,478 in 2000.

 Riverdale is a suburb just south of Hartsfield–Jackson Atlanta International Airport and is part of the Atlanta metropolitan area.
</item 3>

<item 4>
College of Coastal Georgia
The College of Coastal Georgia (also known as CCGA) is a public college located at historical Brunswick, Georgia, United States and the surrounding historical
Golden Isles.

 The College was established in 1961 and opened in 1964, making it one of Georgia's newest state colleges.

 The college transitioned from a two-year community college into a four-year comprehensive institution and conferred its first baccalaureate degrees on May 7,
2011.
</item 4>

<item 5>
Atlanta South Regional Airport
Henry County Airport (FAA LID: KHMP) , is a public-use county airport located three nautical miles (6 km) west of the central business district of Hampton, a
city in Henry County, Georgia, United States.

 It was known as Clayton County Airport – Tara Field, which was the name still used by the Federal Aviation Administration (FAA) and the Georgia Department of
Transportation (GDOT) until approval of the Atlanta South Regional Airport name.

 The airport was renamed Atlanta South Regional Airport, which was approved by the airport board of commissioners in December 2011, and was approved by the GDOT
and the FAA before it went into effect.
</item 5>

<item 6>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 6>

<item 7>
Brunswick Old Town Historic District
Brunswick Old Town Historic District is a historic district in Brunswick, Georgia.

 It was added to the National Register of Historic Places on April 26, 1979 and includes an area bounded by 1st Street, Bay Street, New Bay Street, H Street,
and Cochran Street (4th Ave. and G St., according to one source).

 Brunswick is one of Georgia's two deep-water ports and is the mainland city associated with the Golden Isles of Georgia, at the junction of I-95 and US 82.
</item 7>

<item 8>
Michael Ola
Michael Ola (born April 19, 1988) is an American football offensive tackle for the Buffalo Bills of the National Football League (NFL).

 He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.

 He has also been a member of the Jacksonville Sharks, Montreal Alouettes, Miami Dolphins, Chicago Bears, San Diego Chargers, Detroit Lions, Seattle Seahawks
and New York Giants.
</item 8>

<item 9>
Lake Worth Community High School
Lake Worth Community High School is a B State of Florida rated public high school located in Lake Worth, Florida.

 Established in 1922 as Lake Worth High School, it is currently one of Palm Beach County's largest schools.

 It has six magnet programs: the award-winning Air Force JROTC led by former Multi-National Award-winning Drill Teams, Tyranny and Fidelis; Medicine and Allied
Health; Child Care and Development; Criminal Justice; Culinary and Drafting and Design.

 The Palm Beach County School Board added the word Community to the names of all public high schools, including Lake Worth High School, in the 1980s.
</item 9>

<item 10>
Rich Central High School
Rich Central High School or RCHS is a public four-year high school located in Olympia Fields, Illinois, a southern suburb of Chicago, Illinois, in the United
States.

 The Rich Central Campus serves the cities of Matteson, Richton Park, Country Club Hills, Chicago Heights, Tinley Park, and parts of Olympia Fields.

 It is a part of Rich Township District 227, which also includes Rich East High School and Rich South High School.

 Although the school is located in Olympia Fields, it does not serve the entire village.

 Some Olympia Fields students attend high school at Homewood-Flossmoor High School and Bloom High School, depending on the subdivision where the student
resides.
</item 10>
=== END ITEM id=bdbcad582c7e691e ===

=== ITEM id=6dd169cffa629c51 ===
Gold sentences and facts:
Sentence 0: Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.
  Fact 0.0: Aleksandr Ivanovsky lived from 1881 to 1968.
  Fact 0.1: Aleksandr Ivanovsky was a Russian screenwriter.
  Fact 0.2: Aleksandr Ivanovsky was a Russian film director.
  Fact 0.3: Aleksandr Ivanovsky worked during the Soviet era.
Sentence 1: Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera" (1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".
  Fact 1.0: Arthur Lubin was born on July 25, 1898.
  Fact 1.1: Arthur Lubin died on May 12, 1995.
  Fact 1.2: Arthur Lubin was an American film director.
  Fact 1.3: Arthur Lubin was an American producer.
  Fact 1.4: Arthur Lubin directed several 'Abbott & Costello' films.
  Fact 1.5: Arthur Lubin directed 'Phantom of the Opera', released in 1943.
  Fact 1.6: Arthur Lubin directed the 'Francis the Talking Mule' series.
  Fact 1.7: Arthur Lubin created the talking-horse TV series 'Mister Ed'.

Retrieved items:
<item 1>
Richard Lund
Richard Lund (9 July 1885 – 27 September 1960) was a Swedish film and theatre actor.

 He made his stage debut at Stora Teatern in Gothenburg in 1904 and later appeared in 73 films between 1912 and 1952, making his most important roles during the
silent film era.

 Among his best-known roles is that of Sir Archie in Mauritz Stiller's "Sir Arne's Treasure" (1919).

 He was the most prominent "first lover" of Swedish film during his heyday.
</item 1>

<item 2>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 2>

<item 3>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 3>

<item 4>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 4>

<item 5>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 5>

<item 6>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 6>

<item 7>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 7>

<item 8>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 8>

<item 9>
A Successful Failure
A Successful Failure (1934) is an American film directed by Arthur Lubin.

 It was Lubin's first movie as director.
</item 9>

<item 10>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 10>
=== END ITEM id=6dd169cffa629c51 ===

=== ITEM id=6618b78959991322 ===
Gold sentences and facts:
Sentence 0:  He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems."
  Fact 0.0: The recipient was awarded the 2012 Nobel Prize in Physics.
  Fact 0.1: The recipient shared the 2012 Nobel Prize in Physics jointly with Serge Haroche.
  Fact 0.2: The 2012 Nobel Prize in Physics was awarded for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
Sentence 1:  Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.
  Fact 1.0: Since 2001, Serge Haroche has been a Professor at the Collège de France.
  Fact 1.1: Since 2001, Serge Haroche has held the Chair of Quantum Physics.

Retrieved items:
<item 1>
Nobel Prize in Physics
The Nobel Prize in Physics (Swedish: "Nobelpriset i fysik" ) is a yearly award given by the Royal Swedish Academy of Sciences for those who conferred the most
outstanding contributions for mankind in the field of physics.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel
Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 1>

<item 2>
List of Nobel laureates in Physiology or Medicine
Each recipient receives a medal, a diploma and a monetary award that has varied throughout the years.

 In 1901, von Behring received 150,782 SEK, which is equal to 7,731,004 SEK in December 2008.

 In 2013, the prize was awarded to James E. Rothman, Randy W. Schekman and Thomas C. Südhof; they were recognised "after discovering how cells precisely
transport material".

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 2>

<item 3>
Brian Josephson
Brian David Josephson, {'1': ", '2': ", '3': ", '4': "} (born 4 January 1940), is a Welsh theoretical physicist and professor emeritus of physics at the
University of Cambridge.

 Best known for his pioneering work on superconductivity and quantum tunnelling, he was awarded the Nobel Prize in Physics in 1973 for his prediction of the
Josephson effect, made in 1962 when he was a 22-year-old PhD student at Cambridge University.

 Josephson is the only Welshman to have won a Nobel Prize in Physics.

 He shared the prize with physicists Leo Esaki and Ivar Giaever, who jointly received half the award for their own work on quantum tunnelling.
</item 3>

<item 4>
List of Nobel laureates in Chemistry
The Nobel Prize in Chemistry (Swedish: "Nobelpriset i kemi" ) is awarded annually by the Royal Swedish Academy of Sciences to scientists in the various fields
of chemistry.

 It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, who died in 1896.

 These prizes are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.

 As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Royal
Swedish Academy of Sciences.

 The first Nobel Prize in Chemistry was awarded in 1901 to Jacobus Henricus van 't Hoff, of the Netherlands.

 Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.

 In 1901, van 't Hoff received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 4>

<item 5>
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
</item 5>

<item 6>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 6>

<item 7>
David J. Wineland
David Jeffrey Wineland (born February 24, 1944) is an American Nobel-laureate physicist at the National Institute of Standards and Technology (NIST) physics
laboratory.

 His work has included advances in optics, specifically laser cooling trapped ions and using ions for quantum computing operations.

 He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of
individual quantum systems."
</item 7>

<item 8>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 8>

<item 9>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 9>

<item 10>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 10>
=== END ITEM id=6618b78959991322 ===

=== ITEM id=27729462759ccf98 ===
Gold sentences and facts:
Sentence 0:  He appeared in 57 films between 1919 and 1932.
  Fact 0.0: The person described appeared in 57 films.
  Fact 0.1: The person described appeared in 57 films between 1919 and 1932.
Sentence 1:  He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
  Fact 1.0: The person described directed 54 films.
  Fact 1.1: The person described directed these films between 1920 and 1933.
  Fact 1.2: The person's directed films include one Harry Langdon short.
  Fact 1.3: The person's directed films include two of the Tay Garnett-penned comedies Stan Laurel made for Joe Rock.
  Fact 1.4: The person's directed films include fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
Sentence 2: John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.
  Fact 2.0: John Scott Biddle lived from 1925 to 2008.
  Fact 2.1: John Scott Biddle was a foremost yachting cinematographer.
  Fact 2.2: John Scott Biddle was a lecturer.
  Fact 2.3: John Scott Biddle established a film-making career.
  Fact 2.4: John Scott Biddle's film-making career spanned more than forty years.

Retrieved items:
<item 1>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 1>

<item 2>
John Biddle (Michigan politician)
John Biddle (March 2, 1792 – August 25, 1859) was a delegate to the United States Congress from the Michigan Territory.
</item 2>

<item 3>
John Biddle (Unitarian)
John Biddle or Bidle (born Wotton-under-Edge, Gloucestershire, England, 14 January 1615 – died 22 September 1662) was an influential English nontrinitarian, and
Unitarian.

 He is often called "the Father of English Unitarianism".
</item 3>

<item 4>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 4>

<item 5>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 5>

<item 6>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 6>

<item 7>
Biddle family
The American Biddle family of Philadelphia began when William Biddle (1630–1712) and Sarah Kempe (1634–1709), Quakers, emigrated from England to America in
1681, in part, to avoid religious persecution.

 Having acquired extensive rights to over 43000 acre of lands in Quaker West Jersey, they settled first at Burlington.

 Two third generation brothers, William Biddle, 3rd (1698–1756), and John Biddle (1707–1789), moved from Mount Hope (1684) near Bordentown to Philadelphia in
the 1720s and 1730s and constituted the first generation of the Philadelphia Biddle family.
</item 7>

<item 8>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 8>

<item 9>
Charles John Biddle
Charles John Biddle (April 30, 1819 – September 28, 1873) was an American soldier, lawyer, congressman, and newspaper editor.
</item 9>

<item 10>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 10>
=== END ITEM id=27729462759ccf98 ===
