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

=== ITEM id=d1c1d1ffa8722991 ===
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
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
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
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 3>

<item 4>
Zhao Qinggang
Zhao Qinggang (; born 24 July 1985) is a Chinese track and field athlete who competes in the javelin throw.

 His 89.15 m personal best throw, set in 2014, is the Chinese record.

 It was also the Asian record until surpassed in 2017.

 He represented his country at the 2013 World Championships in Athletics.

 He is the 2014 Asian Games champion.

 At the East Asian Games he was the runner-up in 2009 and winner in 2013.

 He is a two-time Chinese champion (2012 and 2013) and was the winner of the 2013 National Games of China.
</item 4>

<item 5>
Canadian athletes Ross MacKenzie and Brian MacLaren earned 1966 Commonwealth and 1967 Pan American relay silver; both competed in Olympic 400m. Guyanese Harry
Prowell set a marathon record and remains Guyana’s sole Olympic marathoner. Canadian miler Dave Bailey broke four minutes and won Pan Am bronze; Siegmar
Ohlemann won Pan Am 800m silver. Geri Donnelly and China’s Liu Ailing were World Cup footballers; Liu also
</item 5>

<item 6>
The tennis reports mainly track defending doubles champions who withdrew, split partners, or exited early. Chan Yung-jan and Abigail Spears won the 2009 Korea
Open; Sania Mirza and Mahesh Bhupathi withdrew from the 2010 Australian Open mixed doubles, while defending US Open champions Carly Gullickson and Travis
Parrott lost in round one. Francesca Schiavone won the 2010 French Open and reached world No. 4. Lee Chong Wei and
</item 6>

<item 7>
Nicolas N’Godrela won three tennis golds at the 2015 Pacific Games. Aerobic gymnast Remus Nicolai earned 11 world and European medals, then coached in Romania.
Brett McClure won Olympic team silver and is USA Gymnastics’ men’s High Performance Director. Masayoshi Yamamoto won 2014 Asian Games pommel horse gold. Louis
Smith won five Olympic medals across three Games and a 2015 European pommel horse title. Luke
</item 7>

<item 8>
Lee Young-sun
Lee Young-Sun (born 21 February 1974) is a retired female javelin thrower from South Korea.

 Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
</item 8>

<item 9>
Kavita Tungar holds India’s 10 km and half-marathon records; she won 2010 Commonwealth bronze and Asian Games silver, becoming the first Indian woman to win an
individual Commonwealth track medal. Ugandan Evelyn Adiru won the 1982 African 800m title and competed at the 1984 Olympics. Kenyan Mary Chemweno won African
800m and 1500m titles. Indian K. M. Binu holds the 400m national record
</item 9>

<item 10>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 10>
=== END ITEM id=d1c1d1ffa8722991 ===

=== ITEM id=4977b4f8cce1b7b7 ===
Gold sentences and facts:
Sentence 0: Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.
  Fact 0.0: Charles Hastings Judd was a Colonel.
  Fact 0.1: Charles Hastings Judd was born on September 8, 1835.
  Fact 0.2: Charles Hastings Judd died on April 18, 1890.
  Fact 0.3: Charles Hastings Judd was an American businessman.
  Fact 0.4: Charles Hastings Judd was an American rancher.
  Fact 0.5: Charles Hastings Judd was an American courtier.
  Fact 0.6: Charles Hastings Judd was an American politician of the Kingdom of Hawaii.
  Fact 0.7: Charles Hastings Judd was a businessman, rancher, courtier and politician of the Kingdom of Hawaii.
Sentence 1:  He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
  Fact 1.0: Charles Hastings Judd served as Chamberlain to King Kalakaua.
  Fact 1.1: Charles Hastings Judd served as Colonel of the military staff of King Kalakaua.
  Fact 1.2: Charles Hastings Judd traveled with King Kalakaua on his 1881 world tour.
Sentence 2: Kalākaua (November 16, 1836 – January 20, 1891), born David Laʻ amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalākaua and sometimes called The Merrie Monarch, was the last king (and second to last monarch) of the Kingdom of Hawaiʻ i.
  Fact 2.0: Kalakaua was born on November 16, 1836.
  Fact 2.1: Kalakaua died on January 20, 1891.
  Fact 2.2: Kalakaua was born David La'amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalakaua.
  Fact 2.3: Kalakaua is sometimes called The Merrie Monarch.
  Fact 2.4: Kalakaua was the last king of the Kingdom of Hawaii.
  Fact 2.5: Kalakaua was the second-to-last monarch of the Kingdom of Hawaii.

Retrieved items:
<item 1>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 1>

<item 2>
Corps of Military Staff Clerks
The Corps of Military Staff Clerks was an administrative corps of the Canadian Army.

 The Headquarters Corps of Military Staff Clerks was established in 1905.
</item 2>

<item 3>
Arthur Tedder, 1st Baron Tedder
Marshal of the Royal Air Force Arthur William Tedder, 1st Baron Tedder, {'1': ", '2': ", '3': ", '4': "} (11 July 1890 – 3 June 1967) was a senior Royal Air
Force commander.

 He was a pilot and squadron commander in the Royal Flying Corps in the First World War and he went on to serve as a senior officer in the Royal Air Force
during the inter-war years when he served in Turkey, Great Britain and the Far East.

 During the Second World War, as Air Officer Commanding RAF Middle East Command, Tedder directed air operations in the Mediterranean and North Africa, including
the evacuation of Crete and "Operation Crusader" in North Africa.

 His bombing tactics became known as the "Tedder Carpet".

 Later in the war Tedder took command of Mediterranean Air Command and in that role was closely involved in the planning of the Allied invasion of Sicily and
then the Allied invasion of Italy.

 When Operation Overlord—the invasion of France—came to be planned, Tedder was appointed Deputy Supreme Commander at Supreme Headquarters Allied Expeditionary
Force under General Eisenhower.
</item 3>

<item 4>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 4>

<item 5>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 5>

<item 6>
Arthur Tedder, 1st Baron Tedder
After the war he served as Chief of the Air Staff, in which role he advocated increased recruiting in the face of many airmen leaving the service, doubled the
size of RAF Fighter Command and implemented arrangements for the Berlin Airlift in 1948.

 After the war he held senior positions in business and academia.
</item 6>

<item 7>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 7>

<item 8>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 8>

<item 9>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 9>

<item 10>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 10>
=== END ITEM id=4977b4f8cce1b7b7 ===

=== ITEM id=7b16165c3bf11fcf ===
Gold sentences and facts:
Sentence 0: "White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".
  Fact 0.0: 'White People Renovating Houses' is the first episode of the twenty-first season of the American animated television series South Park.
Sentence 1:  It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
  Fact 1.0: 'White People Renovating Houses' is the 278th episode of South Park overall.
  Fact 1.1: 'White People Renovating Houses' first aired on Comedy Central in the United States.
  Fact 1.2: 'White People Renovating Houses' first aired on September 13, 2017.
Sentence 2: The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
  Fact 2.0: The twenty-first season of South Park is an American animated sitcom.
  Fact 2.1: The twenty-first season of South Park premiered on Comedy Central on September 13, 2017.
  Fact 2.2: The twenty-first season of South Park will contain ten episodes.

Retrieved items:
<item 1>
South Park (season 21)
The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
</item 1>

<item 2>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 2>

<item 3>
Holiday Special (South Park)
"Holiday Special" is the third episode in the twenty-first season of the American animated television series "South Park".

 The 280th episode of the series overall, it first aired on Comedy Central in the United States on September 27, 2017.
</item 3>

<item 4>
Put It Down (South Park)
"Put It Down" is the second episode in the twenty-first season of the American animated television series "South Park".

 The 279th episode of the series overall, it first aired on Comedy Central in the United States on September 20, 2017.
</item 4>

<item 5>
“Moho House,” episode 21 of *The Simpsons*’ 28th season and the series’ 617th episode, aired May 7, 2017. The material also surveys the show’s seasons,
episodes, characters, shorts, theme music, and comics. Featured stories include Moe’s stolen Flaming Homer recipe, Mona Simpson’s appearances and death, the
family’s whale rescue, and episodes centered on Lisa, Bart, Homer, and their Springfield neighbors.
</item 5>

<item 6>
The entries cover several *South Park* episodes and seasons: season 21 premiered September 13, 2017, with “White People Renovating Houses,” “Put It Down,” and
“Holiday Special” airing September 13, 20, and 27. Season 2’s “Chickenlover” follows Barbrady learning to read and investigating chicken abuse; “Conjoined Fetus
Lady” features a dodgeball trip to China and a tribute to the school nurse. “Something Wall
</item 6>

<item 7>
South Park (season 10)
The tenth season of "South Park", an American animated television series created by Trey Parker and Matt Stone, began airing on March 22, 2006.

 The tenth season concluded after 14 episodes on November 15, 2006.

 This is the first season in which Kenny does not die and the last season featuring Isaac Hayes (the voice of Chef) as Hayes quit the show following the
backlash behind season nine's "Trapped in the Closet" episode.

 This season also had a minor controversy when the Halloween episode "Hell on Earth 2006" depicted "The Crocodile Hunter's" Steve Irwin with a stingray lodged
in his chest getting thrown out of Satan's Halloween party for not being in costume.

 Episode 2 in this season is the last one with the Braniff Airlines logo.

 All the episodes in this season were written and directed by Trey Parker.
</item 7>

<item 8>
Something Wall-Mart This Way Comes
"Something Wall-Mart This Way Comes" is the ninth episode of the eighth season of the American animated television series "South Park", and the 120th episode of
the series overall.

 It originally aired on Comedy Central in the United States on November 3, 2004.

 In the episode, a Wall-Mart is built in South Park, and the people start to get addicted to shopping from it, due to its irresistibly attractive bargains, thus
leading many businesses in South Park to close down.

 The four boys have to fight against Wall-Mart and to find a way to stop it from taking over the entire town.
</item 8>

<item 9>
White People Renovating Houses
"White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".

 It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
</item 9>

<item 10>
Chickenlover
"Chickenlover" is the fourth episode in the second season of the American animated television series "South Park".

 The 16th episode of the series overall, it originally aired on Comedy Central in the United States on May 27, 1998.

 The episode was written by series co-creators Trey Parker and Matt Stone, along with David R. Goodman, and directed by Parker.

 In the episode, Officer Barbrady resigns as South Park's only police officer because of his illiteracy.

 Anarchy ensues, just as chickens are mysteriously being molested across South Park.

 Barbrady enlists the help of the boys to learn to read and discover who is molesting the chickens.

 Cartman, meanwhile, masquerades as a police officer.
</item 10>
=== END ITEM id=7b16165c3bf11fcf ===

=== ITEM id=a36a22816eae873a ===
Gold sentences and facts:
Sentence 0:  The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and Viv Albertine, who went on to play in The Slits.
  Fact 0.0: The band never played live.
  Fact 0.1: The band never released any recordings.
  Fact 0.2: Like London SS, the band is more famed for the number of band members who later became well known.
  Fact 0.3: Like Masters of the Backside, the band is more famed for the number of band members who later became well known.
  Fact 0.4: Sid Vicious is listed as a band member who later became well known as a member of the Sex Pistols.
  Fact 0.5: Keith Levene is listed as a band member who later became well known.
  Fact 0.6: Keith Levene was an early member of The Clash.
  Fact 0.7: Keith Levene was later a member of Public Image Ltd.
  Fact 0.8: Palmolive is listed as a band member who later became well known.
  Fact 0.9: Viv Albertine is listed as a band member who later became well known.
  Fact 0.10: Viv Albertine went on to play in The Slits.
Sentence 1:  Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
  Fact 1.0: The band was interviewed by a fanzine named SKUM despite never playing live.
  Fact 1.1: In the SKUM interview, Sid Vicious proclaimed 'I'll just be the yob that I am now'.
Sentence 2: Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock band the Sex Pistols.
  Fact 2.0: Sid Vicious was born John Simon Ritchie.
  Fact 2.1: Sid Vicious was born on 10 May 1957.
  Fact 2.2: Sid Vicious died on 2 February 1979.
  Fact 2.3: Sid Vicious was an English bassist.
  Fact 2.4: Sid Vicious was an English vocalist.
  Fact 2.5: Sid Vicious was most famous as a member of the influential punk rock band the Sex Pistols.

Retrieved items:
<item 1>
Nancy Spungen
Nancy Laura Spungen (February 27, 1958 – October 12, 1978) was the American girlfriend of Sex Pistols bassist Sid Vicious and a figure of the 1970s punk rock
scene.

 Spungen's life and death have been the subject of controversy among music historians and fans of the Sex Pistols.
</item 1>

<item 2>
Ex Pistols
The Ex Pistols were a scam act introduced in 1979 by one-time Sex Pistols producer Dave Goodman after his services were substituted for those of professional
producers.

 The Ex Pistols existed as a sound-alike band meant to mislead fans due to the similar music, name, and artwork, that led people to believe their songs were
actually new or lost Sex Pistols recordings.
</item 2>

<item 3>
Bodies (Sex Pistols song)
"Bodies" is a Sex Pistols song about the shock of abortion from the 1977 album "Never Mind the Bollocks, Here's the Sex Pistols".

 The lyrics contain very graphic imagery about a terminated fetus and feature a great deal of profanity for the time: the third and final verse begins with a
couplet in which the word "fuck" is repeated five times in rapid succession.

 Along with the later "Belsen Was a Gas," it is probably the most graphic and controversial Sex Pistols song in both its subject matter and style.

 Musically, it is also the fastest and heaviest song in the Sex Pistols canon — characterized by thudding drums, droning buzzsaw guitar, and shouted vocals.

 As such, it can be considered a significant antecedent to the genres of hardcore, thrash, and speed metal that was to emerge in the 1980s.
</item 3>

<item 4>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 4>

<item 5>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 5>

<item 6>
Revolution in the Classroom
"Revolution in the Classroom" was a single released by the Ex Pistols; despite its packaging, the record has nothing to do with the Sex Pistols.

 The single was released in 1989 on the STP Records label, pressed on various colours of vinyl.

 The song, along with its B-side "Schools Are Prisons", were both included on the Ex-Pistols album, "Deny".

 They also appear disguised as genuine Sex Pistols recordings on two Sex Pistols retrospective albums by former producer Dave Goodman, "Pirates of Destiny" (I
Swirled Records, 1989) and "We Have Cum For Your Children" (Skyclad Records, 1988).

Under the title "Revolution" the song appears on a German Sex Pistols compilation "Anarchy in the USA" from 1992 released on MBC records.

 CD 084-60322 LP 008-60321
</item 6>

<item 7>
Live at Winterland 1978 (Avengers album)
Live at Winterland 1978 is a live album by the Avengers.

 It was released through online music stores on February 16, 2010.

 The album features a recording of their set on January 14, 1978 at the Winterland Ballroom in San Francisco, CA, when they opened for the Sex Pistols on what
would become the Sex Pistols final show, before reuniting years later.

 The Sex Pistols' set has been released on an album of the same name.

 Songs from this album were previously released on a couple 7" bootlegs titled "Penelope" and "Summer of Hate".
</item 7>

<item 8>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 8>

<item 9>
Sid Vicious
Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock
band the Sex Pistols.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=a36a22816eae873a ===

=== ITEM id=f309ec68902bf1ea ===
Gold sentences and facts:
Sentence 0:  Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.
  Fact 0.0: Watkins was born in Des Moines, Iowa.
  Fact 0.1: Watkins rose to fame in the early 1990s.
  Fact 0.2: Watkins rose to fame as a member of the girl-group TLC.
Sentence 1:  She is best known as the former main vocalist and member of South Korean girl group T-ara.
  Fact 1.0: Watkins is best known as the former main vocalist of the South Korean girl group T-ara.
  Fact 1.1: Watkins is best known as the former member of the South Korean girl group T-ara.

Retrieved items:
<item 1>
Rain (Taeyeon song)
"Rain" is a song by South Korean singer Kim Tae-yeon, a member of the South Korean girl group Girls' Generation.

 It was released digitally by S.M. Entertainment on February 3, 2016.
</item 1>

<item 2>
Tionne Watkins
Tionne Tenese Watkins (born April 26, 1970), better known by her stage name T-Boz, is an American singer, songwriter, actress, author, and executive producer.

 Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.

 She has won four Grammy Awards for her work with TLC.
</item 2>

<item 3>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 3>

<item 4>
TLC (group)
TLC is an American girl group whose original line-up consisted of Tionne "T-Boz" Watkins, Lisa "Left Eye" Lopes and Rozonda "Chilli" Thomas.

 Formed in Atlanta, Georgia in 1990, the group was very successful during the 1990s despite numerous spats with the law, each other, and the group's record
label and management.

 They scored nine top-ten hits on the "Billboard" Hot 100, including four number-one singles "Creep", "Waterfalls", "No Scrubs", and "Unpretty".

 The group also recorded four multi-platinum albums, including "CrazySexyCool" (1994) which still remains the only album by a female group to receive a diamond
certification from the Recording Industry Association of America (RIAA).

 TLC also became the first R&B group in history to receive Million certification from the Recording Industry Association of Japan (RIAJ) for "FanMail" (1999).
</item 4>

<item 5>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 5>

<item 6>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 6>

<item 7>
Girl Talk (TLC song)
"Girl Talk" is a song by American group TLC.

 It was written by band members Lisa Lopes and Tionne Watkins along with Anita McLoud, Edmund "Eddie Hustle" Clement, and Kandi Burruss for the group's fourth
studio album, "3D" (2002).

 Featuring production by Hustle, it was released as the album's lead single in September 2002, along with a previously unreleased song "Get Away".

 While the song contains vocals by Lisa Lopes, the music video for "Girl Talk" marked the band's first release without Lopes, following her death in April of
the same year.
</item 7>

<item 8>
Lisa Lopes
Lisa Nicole Lopes (May 27, 1971 – April 25, 2002), better known by her stage name Left Eye, was an American hip hop singer, rapper, actress, songwriter and
producer.

 She rose to fame in the early 1990s as one-third of the girl group TLC, alongside Tionne "T-Boz" Watkins and Rozonda "Chilli" Thomas.

 Besides rapping and singing background vocals on TLC recordings, Lopes was one of the creative forces behind the group.

 She received more cowriting credits than the other members.

 She also designed the outfits and stage for the group and brought concepts to the group image, album titles, artworks and music videos.

 Through her work with TLC, Lopes won four Grammy Awards.
</item 8>

<item 9>
R U the Girl
R U the Girl (also known as R U the Girl with T-Boz and Chilli) is an American reality series that aired on UPN in 2005.

 The series featured Tionne "T-Boz" Watkins and Rozonda "Chilli" Thomas, the remaining members of the all-girl R&B group TLC whose former member, Lisa "Left
Eye" Lopes, died in a car crash in Honduras on April 25, 2002.

 Initially promoted as a contest to permanently replace Lopes 3 years after her death by TLC themselves, both Watkins and Thomas admitted that the winner of the
contest would not be joining TLC full-time and would not be a full-time replacement member; the winner would only provide guest vocals on a new single by the
duo.
</item 9>

<item 10>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 10>
=== END ITEM id=f309ec68902bf1ea ===

=== ITEM id=291f5ff02d641e75 ===
Gold sentences and facts:
Sentence 0:  In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site LiveJasmin.
  Fact 0.0: In the category of pornographic websites, this site was surpassed in the rankings by xHamster.
  Fact 0.1: In the category of pornographic websites, this site was surpassed in the rankings by XVideos.
  Fact 0.2: In the category of pornographic websites, this site was surpassed in the rankings by Pornhub.
  Fact 0.3: In the category of pornographic websites, this site was surpassed in the rankings by the adult webcam site LiveJasmin.
  Fact 0.4: xHamster, XVideos, and Pornhub are competitor sites.
Sentence 1: XVideos is the world's largest pornographic video sharing and viewing website.
  Fact 1.0: XVideos is the world's largest pornographic video sharing and viewing website.

Retrieved items:
<item 1>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 1>

<item 2>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 2>

<item 3>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 3>

<item 4>
YouPorn
YouPorn is a free pornographic video sharing website and one of the 100 most accessed websites in the world.

 Since launching in August 2006, it grew to become the most popular pornographic website on the internet, and, in November 2007, it was reported to be the
largest free pornographic website as well.

 As of February 2013, it was the 83rd most popular website overall and the fifth most popular pornographic website.

 In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site
LiveJasmin.
</item 4>

<item 5>
Pravda.ru
Pravda.ru (formerly Pravda Online) is a Russian internet news website established in 1999 and owned by Pravda.ru Holding (headed by Vadim Gorshenin).
</item 5>

<item 6>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 6>

<item 7>
Pioneer Pekao Investments
Pioneer Pekao is a Polish company managing several investment funds in Poland.

 It is the biggest and first investment funds manager on the Polish market.
</item 7>

<item 8>
XHamster
xHamster is a pornographic media and social networking site headquartered in Limassol, Cyprus.

 xHamster serves user-submitted pornographic videos, webcam models, pornographic photographs, and erotic literature, and incorporates social networking
features.

 xHamster was founded in 2007.

 With more than 10 million members, it is the third most popular pornography website on the Internet after XVideos and Pornhub.
</item 8>

<item 9>
XVideos
XVideos is the world's largest pornographic video sharing and viewing website.

 It's registered to the Polish company, WGCZ Holding.

 Alexa ranks it as the 50th most visited website in the world.
</item 9>

<item 10>
XY TV
It launched in December 2003 and distribution was limited to select cable companies.

 As of May 31, 2006, XY.tv has ceased operations, and is no longer broadcasting.

 It was wholly owned by American radio personality John Garabedian.
</item 10>
=== END ITEM id=291f5ff02d641e75 ===

=== ITEM id=f7d888f8a68d7473 ===
Gold sentences and facts:
Sentence 0: 'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.
  Fact 0.0: A Guarani word, written in the source as 'aba], means 'what gives strength to the Indian'.
  Fact 0.1: The Guarani word is used for the infusions of the bark of a number of trees native to Brazil.
Sentence 1: Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South America that belongs to the Tupi–Guarani family of the Tupian languages.
  Fact 1.0: Guarani specifically refers to the primary variety known as Paraguayan Guarani.
  Fact 1.1: Paraguayan Guarani has the endonym avañe'ẽ, meaning 'the people's language'.
  Fact 1.2: Guarani is an indigenous language of South America.
  Fact 1.3: Guarani belongs to the Tupi–Guarani family of the Tupian languages.

Retrieved items:
<item 1>
Chiribaya Dog
Its original name is unknown.

 (It has been referred to more ambiguously by the term "perro Peruano " or "perro del Perú " ('Peruvian dog', 'dog of Peru'), but this has also been applied to
an extant but ancient hairless variety, referred to in more detail as "perro sin pelo del Perú", 'hairless dog of Peru', or the Peruvian hairless dog, a
favorite in South American dog shows.)
</item 1>

<item 2>
Cora language
Cora is an indigenous language of Mexico of the Uto-Aztecan language family.

 It is spoken by the ethnic group that is widely known as the Cora but who refer to themselves as "Naáyarite".

 The Cora inhabit the northern sierra of the Mexican state Nayarit which is named after its indigenous inhabitants.

 Cora is a Mesoamerican language and shows many of the traits defining the Mesoamerican Linguistic Area.

 Under the General Law of Linguistic Rights of the Indigenous Peoples it is recognized as a "national language" along with 62 other indigenous languages and
Spanish which have the same "validity" in Mexico .
</item 2>

<item 3>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 3>

<item 4>
Achomawi
In anthropology, Achomawi (also Achumawi, Ajumawi and Ahjumawi) are the northerly nine (out of eleven) bands of the Pit River tribe of Native Americans who
lived in what is now northeastern California in the United States.

 These nine autonomous bands (also called "tribelets") of the Pit River Indians spoke various dialects of one common language, and the other two bands spoke
dialects of a related language, called Atsugewi.

 The name "Achomawi" means river dwelling (from ajumma, "river") and the band historically inhabited the Fall River Valley and the Pit River from the south end
of Big Valley Mountains, westerly to Pit River Falls.
</item 4>

<item 5>
Guarani language
Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South
America that belongs to the Tupi–Guarani family of the Tupian languages.

 It is one of the official languages of Paraguay (along with Spanish), where it is spoken by the majority of the population, and where half of the rural
population is monolingual.

 It is spoken by communities in neighboring countries, including parts of northeastern Argentina, southeastern Bolivia and southwestern Brazil, and is a second
official language of the Argentine province of Corrientes since 2004; it is also an official language of Mercosur.
</item 5>

<item 6>
Pagode (music)
Pagode is a type of Brazilian country-folk traditional style of music.

 It's also known as Cipó Preto (black liana), Pagode caipira (rural pagode) or Pagode sertanejo (folk pagode).

 Sertanejo means anything that comes from the back-country, outback or simply the countryside.

 This style of music was pioneered by musicians Tião Carreiro and Lourival dos Santos in the late 1950s, when they fused the "Coco" and "Calango de roda"
rhythms.

 Although other instruments can be added, it is much characterized by its simple, acoustic approach, with the use of just one or two viola guitars, often
accompanied by an acoustic guitar and, at times, a light percussion section.

 A structural feature commonly used in this genre is the alternation of verses sung with the accompaniement of a classical guitar, with solos of viola caipira.

 Pieces can be entirely instrumental or have one or two singers, usually: when there are two lead vocals (that can be men, women or a combination), they sing
together on a steady interval.
</item 6>

<item 7>
Catuaba
The name catuaba (pronounced [ka.two.

'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.

 The most widely used barks are derived from the trees "Trichilia catigua" and "Erythroxylum vaccinifolium".

 Other catuaba preparations use the bark of trees from the following genera or families: "Anemopaegma", "Ilex", "Micropholis", "Phyllanthus", "Secondatia",
"Tetragastris" and species from the Myrtaceae.
</item 7>

<item 8>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 8>

<item 9>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 9>

<item 10>
Indigenous language
An indigenous language or autochthonous language is a language that is native to a region and spoken by indigenous people, often reduced to the status of a
minority language.

 This language would be from a linguistically distinct community that has been settled in the area for many generations.

 Indigenous languages are not necessarily national languages, and the reverse is also true.
</item 10>
=== END ITEM id=f7d888f8a68d7473 ===

=== ITEM id=54a729bd1b17a395 ===
Gold sentences and facts:
Sentence 0: Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story writer.
  Fact 0.0: Doris May Lessing, CH, was born Doris May Tayler (née Tayler).
  Fact 0.1: Doris May Lessing was born on 22 October 1919.
  Fact 0.2: Doris May Lessing died on 17 November 2013.
  Fact 0.3: Doris May Lessing was a British novelist.
  Fact 0.4: Doris May Lessing was a British poet.
  Fact 0.5: Doris May Lessing was a British playwright.
  Fact 0.6: Doris May Lessing was a British librettist.
  Fact 0.7: Doris May Lessing was a British biographer.
  Fact 0.8: Doris May Lessing was a British short story writer.
Sentence 1: Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
  Fact 1.0: Philip Kindred Dick was born on December 16, 1928.
  Fact 1.1: Philip Kindred Dick died on March 2, 1982.
  Fact 1.2: Philip Kindred Dick was an American writer.
  Fact 1.3: Philip Kindred Dick is notable for publishing works of science fiction.

Retrieved items:
<item 1>
The Days of Perky Pat (collection)
The Days of Perky Pat is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Gollancz in 1990 and reprints Volume IV of "The Collected Stories of Philip K. Dick".

 It had not previously been published as a stand-alone volume.

 The stories had originally appeared in the magazines "Galaxy Science Fiction", "Science Fiction Stories", "If", "Fantastic Universe", "Fantasy and Science
Fiction", "Fantastic", "Worlds of Tomorrow", "Escapade" and "Amazing Stories".
</item 1>

<item 2>
Jenny Diski
Jenny Diski FRSL (née Simmonds; 8 July 1947 – 28 April 2016) was an English writer.

 She had a troubled childhood, but was rescued by the older novelist Doris Lessing; she lived in Lessing's house for four years.

 Diski was educated at University College London, and worked as a teacher during the 1970s and early 1980s.
</item 2>

<item 3>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 3>

<item 4>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 4>

<item 5>
Adjustment Team
"Adjustment Team" is a science fiction short story by American writer Philip K. Dick.

 It was first published in "Orbit Science Fiction" (September–October 1954, No. 4) with illustration by Faragasso.

 It was later reprinted in "The Sands of Mars and Other Stories" (Australian) in 1958, "The Book of Philip K. Dick" in 1973, "The Turning Wheel and Other
Stories" (United Kingdom) in 1977, "The Collected Stories of Philip K. Dick" in 1987 (Underwood–Miller), 1988 (Gollancz, United Kingdom), 1990 (Citadel
Twilight, United States), "Selected Stories of Philip K. Dick" in 2002 and in "The Early Work of Philip K. Dick, Volume One: The Variable Man & Other Stories"
in 2009.
</item 5>

<item 6>
Doris Lessing
Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story
writer.

 Her novels include "The Grass Is Singing" (1950), the sequence of five novels collectively called "Children of Violence" (1952–69), "The Golden Notebook"
(1962), "The Good Terrorist" (1985), and five novels collectively known as "Canopus in Argos: Archives" (1979–1983).
</item 6>

<item 7>
The Philip K. Dick Reader
The Philip K. Dick Reader is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Citadel Twilight in 1997.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Imaginative Tales", "Fantastic Universe" and "Space Science
Fiction".

 It is identical in content and order to the edition of volume 3 of the Collected Stories of Philip K. Dick produced by the same publisher apart from the
substitution of three stories in positions 21-23 of 24 and the omission of the end notes in the Collected Stories edition.

 At press time, stories 21 and 24 had already been made into successful movie adaptations and stories 22 and 23 had been optioned.
</item 7>

<item 8>
The Father-Thing (collection)
The Father-Thing is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Gollancz in 1989 and reprints Volume III of "The Collected Stories of Philip K. Dick".

 It had not previously been published as a stand-alone volume.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Science Fiction Quarterly" and "Imaginative Tales".
</item 8>

<item 9>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 9>

<item 10>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 10>
=== END ITEM id=54a729bd1b17a395 ===

=== ITEM id=0cd16053e2b4a612 ===
Gold sentences and facts:
Sentence 0:  It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
  Fact 0.0: The described species is common across temperate parts of Eurasia, northern Africa, and the Americas.
  Fact 0.1: The described species is common across subtropical parts of Eurasia, northern Africa, and the Americas.
  Fact 0.2: The described species is naturalized in Australia.
  Fact 0.3: The described species is naturalized on various islands.
Sentence 1: Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South America.
  Fact 1.0: Echeveria is a large genus of flowering plants.
  Fact 1.1: Echeveria is in the Crassulaceae family.
  Fact 1.2: Echeveria is native to semi-desert areas of Central America.
  Fact 1.3: Echeveria is native to semi-desert areas of Mexico.
  Fact 1.4: Echeveria is native to semi-desert areas of northwestern South America.

Retrieved items:
<item 1>
Echeveria setosa
Echeveria setosa, the Mexican fire cracker, is a species of flowering plant in the Crassulaceae family, native to semi-desert areas of Mexico and common
throughout Puebla.
</item 1>

<item 2>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 2>

<item 3>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 3>

<item 4>
Echeveria elegans
Echeveria elegans (Mexican snow ball, Mexican gem, white Mexican rose) is a species of flowering plant in the Crassulaceae family, native to semi-desert
habitats in Mexico.
</item 4>

<item 5>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 5>

<item 6>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 6>

<item 7>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 7>

<item 8>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 8>

<item 9>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 9>

<item 10>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 10>
=== END ITEM id=0cd16053e2b4a612 ===

=== ITEM id=ebfc7a0c5ed26b37 ===
Gold sentences and facts:
Sentence 0:  In return, William made Copsi earl of Northumbria and sent him back to York.
  Fact 0.0: William made Copsi earl of Northumbria in return.
  Fact 0.1: William sent Copsi back to York.
Sentence 1: York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.
  Fact 1.0: York is a historic walled city.
  Fact 1.1: York is at the confluence of the rivers Ouse and Foss.
  Fact 1.2: York is in North Yorkshire, England.

Retrieved items:
<item 1>
Earl Cowper
He represented Hertford in Parliament.

 He was succeeded by his eldest son, the aforementioned William Cowper, the third Baronet, who was elevated to the peerage as Baron Cowper in 1706 and made Earl
Cowper in 1718.

 In 1706 Lord Cowper married as his second wife Mary Clavering, daughter of John Clavering, of Chopwell, County Durham.
</item 1>

<item 2>
Augustine of Canterbury
Augustine of Canterbury (born first third of the 6th century – died probably 26 May 604) was a Catholic Benedictine monk who became the first Archbishop of
Canterbury in the year 597.

 He is considered the "Apostle to the English" and a founder of the Catholic Church in England.
</item 2>

<item 3>
York Minster Police
York Minster Police is a small, specialised cathedral constabulary responsible for policing York Minster in York, United Kingdom.
</item 3>

<item 4>
Earl of Norbury
Earl of Norbury, in the County of Tipperary, is a title in the Peerage of Ireland.

 It was created in 1827, along with the title Viscount Glandine, of Glandine in the King's County, for the Irish politician and judge John Toler, 1st Baron
Norbury, upon his retirement as Chief Justice of the Common Pleas in Ireland.

 The titles were created with special remainder to his second son, Hector, as his eldest son, Daniel, was then considered mentally unwell.

 Lord Norbury had already been created Baron Norbury, of Ballycrenode in the County of Tipperary, in the Peerage of Ireland in 1800, with remainder to the heirs
male of his body.

 Moreover, his wife, Grace Toler (née Graham), had been created Baroness Norwood, of Knockalton in the County of Tipperary, in the Peerage of Ireland in 1797,
with remainder to the heirs male of her body.

 By the time Lord Norbury was raised to the Earldom, his wife had died and their eldest son had succeeded her as 2nd Baron Norwood.
</item 4>

<item 5>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 5>

<item 6>
Earl Cowper
Earl Cowper (pronounced "Cooper") was a title in the Peerage of Great Britain.

 It was created in 1718 by George I for William Cowper, 1st Baron Cowper, his first Lord Chancellor, with remainder in default of male issue of his own to his
younger brother, Spencer Cowper.

 Cowper had already been created Baron Cowper of Wingham in the County of Kent, in the Peerage of England on 14 December 1706, with normal remainder to the
heirs male of his body, and was made Viscount Fordwich, in the County of Kent, at the same time as he was given the earldom, also Peerage of Great Britain and
with similar remainder.

 He was the great-grandson of William Cowper, who was created a Baronet, of Ratling Court in the County of Kent, in the Baronetage of England on 4 March 1642.

 The latter was succeeded by his grandson, the second Baronet.

 He represented Hertford in Parliament.
</item 6>

<item 7>
Copsi
Copsi (or Copsig; Old English: "Cōpsige" ) [died 1067] was a Northumbrian magnate in late Anglo-Saxon England.

 He was a supporter of Tostig, and was exiled along with him in 1065.

 Copsi soon fled to Orkney (then a part of Norway).

 The next year (1066), he joined Tostig at Sandwich, in Kent, with 17 ships.

 Copsi survived Tostig's defeat at Stamford Bridge, and when William the Conqueror prevailed at Hastings he travelled, in March 1067, to pay William homage at
Barking (where William was staying while his tower was being constructed in London).

 In return, William made Copsi earl of Northumbria and sent him back to York.

 Copsi's rule lasted a mere five weeks, at which time he was murdered by Osulf, son of Eadulf III of the ancient Bernician family which had historically
governed the area from Bamburgh, at Newburn-upon-Tyne.
</item 7>

<item 8>
Copsi
Osulf, however, only ruled as earl until that autumn, when he was killed by an outlaw he was tracking.
</item 8>

<item 9>
Earl of Norbury
This son also succeeded Lord Norbury himself on his death in 1831 as 2nd Baron Norbury, whilst his younger brother Daniel succeeded to the viscountcy and
earldom according to the special remainder.

 In 1832, the second Earl also succeeded his elder brother in the two baronies.

 He had already in 1825 assumed the additional surname of Graham by Royal licence.
</item 9>

<item 10>
York
York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.

 The municipality is the traditional county town of the historic county of Yorkshire to which it gives its name.

 The city has a rich heritage and has provided the backdrop to major political events in England throughout much of its two millennia of existence.

 The city offers a wealth of historic attractions, of which York Minster is the most prominent, and a variety of cultural and sporting activities making it a
popular tourist destination for millions.
</item 10>
=== END ITEM id=ebfc7a0c5ed26b37 ===

=== ITEM id=1a9a4822646d0147 ===
Gold sentences and facts:
Sentence 0: Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.
  Fact 0.0: Talking Heads were an American rock band.
  Fact 0.1: Talking Heads were formed in 1975.
  Fact 0.2: Talking Heads were formed in New York City.
  Fact 0.3: Talking Heads were active until 1991.
Sentence 1:  is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
  Fact 1.0: The group is a surf rock group.
  Fact 1.1: The group formed in Auburn, Alabama.
  Fact 1.2: The group formed in the early 1990s.
  Fact 1.3: The group came to prominence over the following decade.

Retrieved items:
<item 1>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 1>

<item 2>
No Talking, Just Head
No Talking, Just Head is an album released in 1996 by The Heads, a band composed of Jerry Harrison, Tina Weymouth, and Chris Frantz of Talking Heads, joined by
a variety of guest singers.

 Its name may be seen as an allusion to the fact that Talking Heads' former vocalist, David Byrne, is the only member not involved.
</item 2>

<item 3>
The Best of Talking Heads
The Best of Talking Heads is a 2004 greatest hits album by Talking Heads, released by Sire/Rhino/Warner Bros., and contains in all 18 tracks, from the beginning
to the end of Talking Heads' history.

 It was released the same day (August 17, 2004) as the expanded reissue of "The Name of This Band Is Talking Heads".

 The album charted at number 87 on the ARIA Charts and charted at number 96 on the Ultratop Charts in Belgium.
</item 3>

<item 4>
Man or Astro-man?
Man or Astro-man?

 is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
</item 4>

<item 5>
More Songs About Buildings and Food
More Songs About Buildings and Food is the second studio album by the American rock band Talking Heads, released in July 1978.

 The album was the first of three Talking Heads LPs produced by collaborator Brian Eno.

 It saw the group move musically toward a danceable style, crossing singer David Byrne's unusual delivery with new emphasis on the rhythm section (made up of
bassist Tina Weymouth and drummer Chris Frantz).
</item 5>

<item 6>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 6>

<item 7>
The Happenings
The Happenings are a pop music group that originated in the 1960s.

 Members of the original group, created in the spring of 1961 and initially called "The Four Grads" because all had just graduated from high school in Paterson,
New Jersey, were Bob Miranda, David Libert, Tom Giuliano, and Ralph DiVito.

 In 1968 DiVito was replaced by Bernie LaPorta and Lenny Conforti also joined to play drums in the touring band.

 Both LaPorta and Conforti took a hiatus from the northern New Jersey band, The Emerald Experience, to play and tour with The Happenings.

 That lineup performed mostly at colleges and universities until 1970, when Libert left the band to manage other groups, including George Clinton and
Parliament-Funkadelic, Living Colour, Brian Auger, Vanilla Fudge, The Runaways (Cherie Currie, Joan Jett, Lita Ford), Mother's Finest, Alice Cooper and Evelyn
"Champagne" King.
</item 7>

<item 8>
Talking Heads Africa
Talking Heads (Africa) was introduced in Cape Town in 2008 as part of the Infecting the City public art festival.

 Talking Heads has four core components that form the project.

 These include: developing a platform for conversation and exchange with and between experts; creating a network of African thought leaders; shooting
mini-documentaries that define these leaders and their contributions; developing the tools to make this model work in cities all over the African continent.
</item 8>

<item 9>
Talking Heads
Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.

 The band comprised David Byrne (lead vocals, guitar), Chris Frantz (drums), Tina Weymouth (bass), and Jerry Harrison (keyboards, guitar).

 Described by critic Stephen Thomas Erlewine as "one of the most critically acclaimed bands of the '80s," the group helped to pioneer new wave music by
integrating elements of punk, art rock, funk, and world music with avant-garde sensibilities and an anxious, clean-cut image.
</item 9>

<item 10>
Method Man
Clifford Smith (born April 1, 1971), better known by his stage name Method Man, is an American rapper, record producer, and actor.

 He is known as a member of the East Coast hip hop collective Wu-Tang Clan.

 He is also one half of the hip hop duo Method Man & Redman.

 He took his stage name from the 1979 film "Method Man".

 In 1996, he won a Grammy Award for Best Rap Performance by a Duo or Group, for "I'll Be There for You/You're All I Need to Get By", with American R&B
singer-songwriter Mary J. Blige.
</item 10>
=== END ITEM id=1a9a4822646d0147 ===

=== ITEM id=4fb24f235d0659eb ===
Gold sentences and facts:
Sentence 0: Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's World Cup.
  Fact 0.0: Liu Ailing was born on June 2, 1967.
  Fact 0.1: Liu Ailing is a Chinese former footballer.
  Fact 0.2: Liu Ailing played for the China national team at the 1991 FIFA Women's World Cup.
  Fact 0.3: Liu Ailing played for the China national team at the 1995 FIFA Women's World Cup.
  Fact 0.4: Liu Ailing played for the China national team at the 1999 FIFA Women's World Cup.
Sentence 1: The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale de Football Association" (FIFA), the sport's global governing body.
  Fact 1.0: The FIFA Women's World Cup is an international football competition.
  Fact 1.1: The FIFA Women's World Cup is contested by the senior women's national teams of the members of FIFA.
  Fact 1.2: FIFA stands for Fédération Internationale de Football Association.
  Fact 1.3: FIFA is the sport's global governing body.

Retrieved items:
<item 1>
Liu Ailing
Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's
World Cup.

 She won a silver medal at the 1996 Atlanta Olympics and participated at the 2000 Sydney Olympics.

 A playmaking midfielder, she played professional club football in Japan and the United States.
</item 1>

<item 2>
EAFF E-1 Football Championship
EAFF E-1 Football Championship, known as the East Asian Football Championship from 2003 to 2010, and the EAFF East Asian Cup for the 2013 and 2015 editions, is
a men's international football competition in East Asia for member nations of the East Asian Football Federation (EAFF).

 Before the EAFF was founded in 2002, the Dynasty Cup was held between the East Asian top four teams, and was regarded as the unofficial East Asian
Championship.

 There is a separate competition for both men (first held in 2003) and women (first held in 2005).

 There was also a combined points competition in 2005, where the results of the men's and women's teams are added together (not including qualifiers).
</item 2>

<item 3>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 3>

<item 4>
EAFF E-1 Football Championship (women)
EAFF E-1 Football Championship is an international football competition in East Asia for national teams of the East Asian Football Federation (EAFF).

 The competition between women's national teams is held alongside men's competition.
</item 4>

<item 5>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 5>

<item 6>
Women's Nordic Football Championship
Women's Nordic Football Championship was an international football competition contested by the women's national football teams of the Nordic countries.

 The tournament was held annually between 1974 and 1982.

 Finland, Denmark and Sweden competed from the start, Norway joined the tournament in 1978.

 Iceland and Faroe Islands did not take part at the competition.
</item 6>

<item 7>
Geri Donnelly
Geraldine "Geri" Donnelly (born 30 November 1965) is a Canadian former soccer player.

 A midfielder, she represented Canada at the 1995 and 1999 editions of the FIFA Women's World Cup and was named Canadian Player of the Year in 1996 and 1999.

 Donnelly was part of the Canadian squad who won the 1998 CONCACAF Women's Championship.

 She was selected as a member of the Canadian Soccer Hall of Fame in 2014.
</item 7>

<item 8>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 8>

<item 9>
Gambia women's national football team
The Gambia women's national football team represents the Gambia in international football competition.

 The team, however, has not competed in a match recognised by FIFA, the sport's international governing body, despite that organised women's football has been
played in the country since 1998.

 The Gambia has two youth teams, an under-17 side that has competed in FIFA U-17 Women's World Cup qualifiers, and an under-19 side that withdrew from regional
qualifiers for an under-19 World Cup.

 The development of a national team faces challenges similar to those across Africa, although the national football association has four staff members focusing
on women's football.
</item 9>

<item 10>
FIFA Women's World Cup
The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale
de Football Association" (FIFA), the sport's global governing body.

 The competition has been held every four years since 1991, when the inaugural tournament, then called the Women's World Championship, was held in China.
</item 10>
=== END ITEM id=4fb24f235d0659eb ===
