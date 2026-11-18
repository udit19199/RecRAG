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

=== ITEM id=eb9ecc65ac265e77 ===
Gold sentences and facts:
Sentence 0: The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR: "Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet Union) and South Korea (with the principal support of the United States).
  Fact 0.0: The Korean War is called 한국전쟁 in South Korean Hangul.
  Fact 0.1: The Korean War is called 韓國戰爭 in South Korean Hanja.
  Fact 0.2: The Korean War is romanized as 'Hanguk Jeonjaeng' (RR), meaning 'Korean War'.
  Fact 0.3: The Korean War is called 조국해방전쟁 in North Korean Chosŏn'gŭl.
  Fact 0.4: The Korean War is called 祖國解放戰爭 in North Korean Hancha.
  Fact 0.5: The Korean War is romanized as 'Choguk haebang chǒnjaeng' (MR), meaning 'Fatherland Liberation War'.
  Fact 0.6: The Korean War lasted from 25 June 1950 to 27 July 1953.
  Fact 0.7: The Korean War was a war between North Korea and South Korea.
  Fact 0.8: North Korea fought with the support of China and the Soviet Union.
  Fact 0.9: South Korea fought with the principal support of the United States.
Sentence 1: The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.
  Fact 1.0: The Western Allied invasion of Germany was coordinated by the Western Allies.
  Fact 1.1: The Western Allied invasion of Germany took place during the final months of hostilities in the European theatre of World War II.
Sentence 2:  The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.
  Fact 2.0: The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945.
  Fact 2.1: After crossing the Rhine, the Western Allies fanned out and overran all of western Germany.
  Fact 2.2: The Western Allies overran western Germany from the Baltic in the north to Austria in the south.
  Fact 2.3: The Germans surrendered on 8 May 1945.

Retrieved items:
<item 1>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 1>

<item 2>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 2>

<item 3>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 3>

<item 4>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 4>

<item 5>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 5>

<item 6>
Allied invasion of Italy
The Allied invasion of Italy was the Allied amphibious landing on mainland Italy that took place on 3 September 1943 during the early stages of the Italian
Campaign of World War II.

 The operation was undertaken by General Sir Harold Alexander's 15th Army Group (comprising General Mark W. Clark and General Bernard Montgomery's British
Eighth Army) and followed the successful invasion of Sicily.

 The main invasion force landed around Salerno on 9 September on the western coast in Operation Avalanche, while two supporting operations took place in
Calabria (Operation Baytown) and Taranto (Operation Slapstick).
</item 6>

<item 7>
South German Offensive
The South German Offensive is the general name of one of the final offensives of World War II in Europe.

 The offensive was led by the Seventh and Third armies of the United States along with the First Army of France.

 Soviet troops linked up with American forces in Czechoslovakia notably in the Battle of Slivice.

 The offensive was made by the US 6th Army Group to protect the US 12th Army Group's right flank and to prevent a German last stand in the Alps.

 However German resistance was much more fierce than in the north, which slowed the 6th Army Group's progress.

 However, by the end of April, many German divisions surrendered without a fight to the advancing American forces to avoid the inevitable destruction.

 The VI Corps of the Seventh Army linked up with the US Fifth Army, which fought through Italy, in the Alps as the Third Army advanced into Austria and
Czechoslovakia, where it linked up with Soviet forces advancing from the east.

 Fighting continued a few days after the Surrender of Germany on 8 May, due to German forces fighting west to surrender to the Americans instead of the Soviets.
</item 7>

<item 8>
Schlieffen Plan
The Schlieffen Plan (German: "Schlieffen-Plan" , ] ) was the name given after World War I to the thinking behind the German invasion of France and Belgium on 4
August 1914.

 Field Marshal Alfred von Schlieffen, the Chief of the Imperial Army German General Staff from 1891 to 1906, devised a deployment plan for a war-winning
offensive, in a one-front war against the French Third Republic from 1905–06.

 After the war, the German official historians of the "Reichsarchiv" and other writers, described the plan as a blueprint for victory.

 German historians claimed that the plan had been ruined by "Generaloberst" (Colonel-General) Helmuth von Moltke the Younger, the Commander-in-Chief of the
German army after Schlieffen retired in 1906, who was dismissed after the First Battle of the Marne (5–12 September 1914).
</item 8>

<item 9>
Western Allied invasion of Germany
The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.

 The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany
from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.

 This is known as the "Central Europe Campaign" in United States military histories.
</item 9>

<item 10>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 10>
=== END ITEM id=eb9ecc65ac265e77 ===

=== ITEM id=23db475a6bf69033 ===
Gold sentences and facts:
Sentence 0:  She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".
  Fact 0.0: The person is a member of the Game of Thrones stunt teams.
  Fact 0.1: The Game of Thrones stunt teams won two Screen Actors Guild Awards.
  Fact 0.2: The person has done stunt performances for films including Rogue One.
Sentence 1:  "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
  Fact 1.0: Rogue One follows a group of rebels.
  Fact 1.1: The rebels are on a mission to steal the plans for the Death Star.
  Fact 1.2: The Death Star is the Galactic Empire's superweapon.

Retrieved items:
<item 1>
Race to Nowhere
Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
</item 1>

<item 2>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 2>

<item 3>
The Runaways (film)
The Runaways is a 2010 American drama film about the 1970s rock band of the same name written and directed by Floria Sigismondi.

 It is based on the book "Neon Angel: A Memoir of a Runaway" by the band's original lead vocalist Cherie Currie.

 The film stars Dakota Fanning as Currie, Kristen Stewart as rhythm guitarist and vocalist Joan Jett, and Michael Shannon as record producer Kim Fowley.

 "The Runaways" depicts the formation of the band in 1975 and focuses on the relationship between Currie and Jett until Currie's departure from the band.

 The film grossed about $4.7 million worldwide and received generally favorable reviews from critics.
</item 3>

<item 4>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 4>

<item 5>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 5>

<item 6>
Frenemies (film)
Frenemies is a 2012 teen comedy-drama television film and anthology based on the novel of the same name by Alexa Young which premiered on Disney Channel.

 It features an ensemble cast starring Bella Thorne, Zendaya, Stefanie Scott, Nick Robinson, Mary Mouser and features Connor Price, Jascha Washington and Dylan
Everett.

 The film follows three pairs of teenage friends that go from friends to enemies and back again.

 The film was directed by Daisy Mayer and written by Dava Savel, Wendy Weiner, and Jim Krieg.

 The Disney Channel Original Movie premiered on January 13, 2012 in the United States and Canada.
</item 6>

<item 7>
Rogue One
Rogue One: A Star Wars Story, or simply Rogue One, is a 2016 American epic space opera film directed by Gareth Edwards.

 The screenplay by Chris Weitz and Tony Gilroy is from a story by John Knoll and Gary Whitta.

 It was produced by Lucasfilm and distributed by Walt Disney Studios Motion Pictures.

 It is the first installment of the "Star Wars" "Anthology" series, set immediately before the events of the original "Star Wars" film.

 The cast includes Felicity Jones, Diego Luna, Riz Ahmed, Ben Mendelsohn, Donnie Yen, Mads Mikkelsen, Alan Tudyk, Jiang Wen and Forest Whitaker.

 "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
</item 7>

<item 8>
Rachelle Beinart
Rachelle Amy Beinart (born 29 September 1984) is a British actress and stunt performer.

 She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".

 She has also acted in preschool children's shows as a suit actor, including the reboot of "Teletubbies".
</item 8>

<item 9>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 9>

<item 10>
Rachelle Henry
Rachelle Henry (born December 16, 2000) is an American teen actress and filmmaker.

 She played the role of Sandy Hobbs in the TLC (TV network) Series "Escaping the Prophet"and Lissa Golaski in "Depth", the film prequel to the Soma (video game)
by Frictional Games.

 She is also known for directing and producing short films containing messages of social influence and coming of age themes including "MISSING" and "Almost
Boyfriends".
</item 10>
=== END ITEM id=23db475a6bf69033 ===

=== ITEM id=ddc9ecabeb1ebbfe ===
Gold sentences and facts:
Sentence 0: Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry, famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
  Fact 0.0: Aram Saroyan was born on September 25, 1943.
  Fact 0.1: Aram Saroyan is an American poet.
  Fact 0.2: Aram Saroyan is an American novelist.
  Fact 0.3: Aram Saroyan is an American biographer.
  Fact 0.4: Aram Saroyan is an American memoirist.
  Fact 0.5: Aram Saroyan is an American playwright.
  Fact 0.6: Aram Saroyan is especially known for his minimalist poetry.
  Fact 0.7: A famous example of Aram Saroyan's minimalist poetry is the one-word poem "lighght".
  Fact 0.8: A famous example of Aram Saroyan's minimalist poetry is a one-letter poem comprising a four-legged version of the letter "m".
Sentence 1: Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of microelectronics.
  Fact 1.0: Dr. Edward Keonjian was born on 14 August 1909.
  Fact 1.1: Dr. Edward Keonjian died on 6 September 1999.
  Fact 1.2: Dr. Edward Keonjian was a prominent engineer.
  Fact 1.3: Dr. Edward Keonjian was an early leader in the field of low-power electronics.
  Fact 1.4: Dr. Edward Keonjian was the father of microelectronics.

Retrieved items:
<item 1>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 1>

<item 2>
Aram Saroyan
Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry,
famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
</item 2>

<item 3>
Edward Keonjian
Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of
microelectronics.

 In 1954 Keonjian designed the world's first solar-powered, pocket-sized radio transmitter.

 In 1959 Keonjian designed the first prototype of integrated circuit.

 In 1963 he organized the world's first international symposium on low-power electronics.

 Later on Keonjian collaborated with NASA astronaut Neil Armstrong as chief of failure analysis on the Apollo 11 project.
</item 3>

<item 4>
Lord Cardross
Lord Cardross is a title in the Peerage of Scotland, since 1695 a subsidiary title of the earldom of Buchan.

 It was created in 1606 for John Erskine, 18th Earl of Mar, with remainder to his heirs male and assignees whatsoever and with the power to nominate his
successor.

 In 1617 he nominated his second son by his second wife, Henry Erskine, Master of Cardross, to be his successor in the lordship of Cardross.

 The Earl of Mar died in 1634 and was succeeded in the earldom of Mar by his son by his first wife, John, and in the lordship of Cardross by his grandson David
Erskine, the second Lord Cardross, the son of Henry, Master of Cardross, who had died in 1628.

 The second Lord was a supporter of The Engagement and was barred from sitting in Parliament in 1649.

 He was succeeded by his eldest son, the third Lord.

 He emigrated to North America to escape religious persecution and established a colony in what is now Carolina.
</item 4>

<item 5>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 5>

<item 6>
Strawberry Saroyan
Strawberry Saroyan (born 1970) is a journalist and author.

 The daughter of Aram Saroyan and granddaughter of playwright William Saroyan and actress Carol Matthau, she spent her childhood in Bolinas, California.

 She has a sister named Cream.

 She writes for the New York Times Style section and is the author of "".
</item 6>

<item 7>
My Name Is Aram
My Name is Aram is a book of short stories by William Saroyan first published in 1940.

 The stories detail the exploits of Aram Garoghlanian, a boy of Armenian descent growing up in Fresno, California, and the various members of his large family.

 This book is assigned reading in some schools.
</item 7>

<item 8>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 8>

<item 9>
Levan of Imereti
Levan (Georgian: ლევანი ) (1573–1590), of the Bagrationi Dynasty, was a king of Imereti from 1585 to 1588.

 He succeeded on the death of his father, George II, in 1585 when he was twelve years old.

 With his ascend to the throne, Leon faced a revolt by his own uncle, Constantine, who defied the royal authority and took control of Upper Imereti.

 Leon made an alliance with the Mingrelian prince Mamia IV Dadiani, married his sister Marekhi, and forced Constantine to surrender in 1587.

 A year later, Imereti was invaded by Simon I, the resurgent king of Kartli in eastern Georgia, who sought to reunify all Georgian lands under his crown.

 Leon was forced to flee to the highland province of Lechkhumi, but was soon able to resume the throne after Simon had to return to Kartli.
</item 9>

<item 10>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 10>
=== END ITEM id=ddc9ecabeb1ebbfe ===

=== ITEM id=589c1eac6ac3bfe4 ===
Gold sentences and facts:
Sentence 0: The City Heiress is a play by Aphra Behn produced in 1682.
  Fact 0.0: The City Heiress is a play by Aphra Behn.
  Fact 0.1: The City Heiress was produced in 1682.
Sentence 1: Aphra Behn ( ; 14 December 1640?
  Fact 1.0: Aphra Behn's birth date is listed as 14 December 1640, marked with a question mark as uncertain.
Sentence 2:  (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.
  Fact 2.0: The 14 December 1640 date for Aphra Behn is a baptismal date.
  Fact 2.1: Aphra Behn died on 16 April 1689.
  Fact 2.2: Aphra Behn was a British playwright.
  Fact 2.3: Aphra Behn was a poet.
  Fact 2.4: Aphra Behn was a translator.
  Fact 2.5: Aphra Behn was a fiction writer.
  Fact 2.6: Aphra Behn was from the Restoration era.

Retrieved items:
<item 1>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 1>

<item 2>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 2>

<item 3>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 3>

<item 4>
Elswyth Thane
Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.

 Born in Burlington, Iowa, she was the daughter of a local teacher and high school principal.

 The family moved to New York City in 1918, and "Helen Ricker" changed her name to "Elswyth Thane".

 She began working as a freelance writer in the 20s, and became a newspaper writer and a Hollywood screenwriter.

 Her first novel, "Riders of the Wind", was published in 1926.

 Her novel, "The Tudor Wench", about Elizabeth I of England, was made into a play.

 She was a collector of scarves.
</item 4>

<item 5>
The Heiress (1786 play)
The Heiress is a comedy play by the British playwright and soldier John Burgoyne.

 The play debuted at the Drury Lane Theatre on 14 January 1786.

 It concerns the engagement of Lord Gayville to Miss Alscrip, a fashionable woman he believes to be an heiress.

 Gayville later discovers that the woman who really stands to inherit the fortune is his true love Miss Clifford.

 The play was an enormous success, running for 31 performances in its initial season and being revived again the following year.

 The play was initially anonymous, but Burgoyne was soon widely reported to be the author and he acknowledged this after the play's debut.
</item 5>

<item 6>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 6>

<item 7>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 7>

<item 8>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 8>

<item 9>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 9>

<item 10>
The City Heiress
The City Heiress is a play by Aphra Behn produced in 1682.

 The play conforms to the general rules of Restoration comedy, but it also keeps Behn's own highly Royalist political point of view.
</item 10>
=== END ITEM id=589c1eac6ac3bfe4 ===

=== ITEM id=c18ae5dc39da1f59 ===
Gold sentences and facts:
Sentence 0: Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.
  Fact 0.0: Cherie Ann Currie was born November 30, 1959.
  Fact 0.1: Cherie Ann Currie is an American musician.
  Fact 0.2: Cherie Ann Currie is an instrumentalist, singer, and songwriter.
  Fact 0.3: Cherie Ann Currie is an actress.
  Fact 0.4: Cherie Ann Currie is an artist.
Sentence 1: Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
  Fact 1.0: Steve Augeri was born January 30, 1959.
  Fact 1.1: Steve Augeri is an American rock singer.
  Fact 1.2: Steve Augeri is best known for his work with Tall Stories.
  Fact 1.3: Steve Augeri is best known for his work with Tyketto.
  Fact 1.4: Steve Augeri is best known for his work with Journey.

Retrieved items:
<item 1>
Messin' with the Boys
Messin' with the Boys is the second post-Runaways album by Cherie Currie, released in 1980 For this album Currie worked with her identical twin sister, Marie
Currie.

 Their band was called "Cherie and Marie Currie".

 Marie was a guest vocalist on Cherie's first album, so Marie went on tour with Cherie to support her first album, "Beauty's Only Skin Deep".

 When Marie would join Cherie on stage to sing the encores the audience would go wild.

 So Cherie ran with the idea of two blonds are better than one.

 The idea paid off because "Messin' with the Boys" received more radio play than "Beauty's Only Skin Deep", and their song "Since You Been Gone" made it to 95
on US charts.

 The single "This Time" and the album Messin' with the Boys made the top 200 on U.S. charts.

 This makes Messin' with the Boys Cherie Currie's most successful album.

 "I Just Love the Feeling" originally surfaced on the 1974 album, "S.S.
</item 1>

<item 2>
The Runaways (film)
The Runaways is a 2010 American drama film about the 1970s rock band of the same name written and directed by Floria Sigismondi.

 It is based on the book "Neon Angel: A Memoir of a Runaway" by the band's original lead vocalist Cherie Currie.

 The film stars Dakota Fanning as Currie, Kristen Stewart as rhythm guitarist and vocalist Joan Jett, and Michael Shannon as record producer Kim Fowley.

 "The Runaways" depicts the formation of the band in 1975 and focuses on the relationship between Currie and Jett until Currie's departure from the band.

 The film grossed about $4.7 million worldwide and received generally favorable reviews from critics.
</item 2>

<item 3>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 3>

<item 4>
Marie Currie
Marie Michelle Currie (born November 30, 1959) is an American singer, songwriter, actress, and artist.

 Currie is best known for playing in a band with her identical twin Cherie Currie called Cherie & Marie Currie.

 Their song "Since You Been Gone" charted at number 95 on the US charts.

 Marie played Singing Maid Marie in "The Rosebud Beach Hotel" and is now a popular multi-media sculptress and artist.
</item 4>

<item 5>
Berlin (band)
Berlin is an American new wave band.

 The group was formed in Orange County in 1979 by John Crawford (bass guitar).

 Band members included Crawford, Terri Nunn (vocals), David Diamond (keyboards), Ric Olsen (guitar), Matt Reid (keyboards) and Rod Learned (drums).

 The band gained mainstream-commercial success in the early 1980s with singles including "The Metro", "Sex (I'm A..

.)

", "No More Words" and then in the mid 1980s with the chart-topping single "Take My Breath Away" from the 1986 film "Top Gun".
</item 5>

<item 6>
Reverie (Cherie Currie album)
Reverie is the third full-length studio album by Cherie Currie.

 Released on iTunes March 16, 2015.

 Cherie released the CD version of this album June 5, 2015 on her eBay page cheriecurriedirect.

 There is a 35-year gap between Cherie's last full-length studio album, 1980's Messin' with the Boys (with Marie Currie), and 2015's Reverie.

 This is last studio album Kim Fowley produced before his death.

 Kim helped Cherie release this album to make amends with her after all the money he swindled her out of when she was in the Runaways and for releasing her and
Marie's music on Young and Wild without their approval.

 After Kim's death Cherie's son, Jake Hays, took over producing.
</item 6>

<item 7>
Steve Augeri
Steve Augeri (born January 30, 1959) is an American rock singer best known for his work with Tall Stories, Tyketto, and Journey.
</item 7>

<item 8>
Young and Wild (album)
Young and Wild is a compilation by Cherie & Marie Currie.

 This album has all 10 original tracks from Messin' with the Boys, six songs from Beauty's Only Skin Deep, three songs Cherie Currie sang with The Runaways, and
one new track co-written by Marie Currie, "Longer Than Forever".

 "Longer Than Forever" was the B side of the single "Since You Been Gone".
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 10>
=== END ITEM id=c18ae5dc39da1f59 ===

=== ITEM id=36412ca2b21b4fad ===
Gold sentences and facts:
Sentence 0: The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
  Fact 0.0: The Consul is an opera.
  Fact 0.1: The Consul is an opera in three acts.
  Fact 0.2: The Consul has music by Gian Carlo Menotti.
  Fact 0.3: The Consul has a libretto by Gian Carlo Menotti.
  Fact 0.4: The Consul is Gian Carlo Menotti's first full-length opera.
Sentence 1:  He completed the music for the opera while living in Zurich in 1916.
  Fact 1.0: Gian Carlo Menotti completed the music for The Consul.
  Fact 1.1: Gian Carlo Menotti completed the music for The Consul in 1916.
  Fact 1.2: Gian Carlo Menotti was living in Zurich in 1916.
  Fact 1.3: Gian Carlo Menotti completed the music for The Consul while living in Zurich.

Retrieved items:
<item 1>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 1>

<item 2>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 2>

<item 3>
Arlecchino (opera)
Arlecchino, "oder Die Fenster" ("Harlequin, or The Windows", is a one-act opera with spoken dialog by Ferruccio Busoni, with a libretto in German, composed in
1913.

 He completed the music for the opera while living in Zurich in 1916.

 It is a number opera written in neo-classical style and includes ironic allusions to operatic conventions and situations typical of the late 18th and early
19th centuries.

 It even includes a parody of a duel.
</item 3>

<item 4>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 4>

<item 5>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 5>

<item 6>
The Consul
The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
</item 6>

<item 7>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 7>

<item 8>
Columbina
Columbina (in Italian Colombina, meaning "little dove"; in French and English Colombine) is a stock character in the "Commedia dell'Arte".

 She is Harlequin's mistress, a comic servant playing the tricky slave type, and wife of Pierrot.

 Rudlin and Crick use the Italian spelling Colombina in "Commedia dell'arte: A Handbook for Troupes".
</item 8>

<item 9>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 9>

<item 10>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 10>
=== END ITEM id=36412ca2b21b4fad ===

=== ITEM id=ba78999987ec623f ===
Gold sentences and facts:
Sentence 0: The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical creature of the forest, or Rå, in Swedish folklore.
  Fact 0.0: The skogsrå is also called Forest Rå.
  Fact 0.1: The skogsrå is also called Skogsfrun (The Mistress of the Forest).
  Fact 0.2: The skogsrå is also called Skogssnuvan.
  Fact 0.3: The skogsrå is also called Skogsnymfen (Forest Nymph).
  Fact 0.4: The skogsrå is also called Råndan (the Rå).
  Fact 0.5: The skogsrå is also called huldra.
  Fact 0.6: The skogsrå was a mythical creature of the forest, also called Rå, in Swedish folklore.
Sentence 1: A hulder is a seductive forest creature found in Scandinavian folklore.
  Fact 1.0: A hulder is a seductive forest creature.
  Fact 1.1: Hulder creatures are found in Scandinavian folklore.

Retrieved items:
<item 1>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 1>

<item 2>
Jenny Nyström
Jenny Eugenia Nyström (13 or 15 June 1854 in Kalmar, Sweden – 17 January 1946 in Stockholm) was a painter and illustrator who is mainly known as the person who
created the Swedish image of the "jultomte" on numerous Christmas cards and magazine covers, thus linking the Swedish version of Santa Claus to the gnomes of
Scandinavian folklore.
</item 2>

<item 3>
Selkie (disambiguation)
A selkie is a mythological creature found in Faroese, Icelandic, Irish, and Scottish folklore.
</item 3>

<item 4>
Almaz (mythical beast)
The almaz (Chechen: "almaz" or "hun-sag"; Adyghe: "almesti"), roughly translated as various "feral forest-man" or "stone spirit", is a mythical beast that is
considered to be an evil forest creature with magical powers residing in its hair that exists in Chechen and Ingush folk beliefs and Circassian folk beliefs.

 The first "attestation" of it in writing was by a Bavarian captive of the Mongols, but it is present in the national folklore of Chechens and Ingush and of
Circassians.

 The male almaz is said to be hairy and hideous, and have an axe embedded in its chest, while the female is very beautiful with large breasts and golden hair,
and has a "favorite pastime" of dancing naked at night under the moon.

 The almaz is said to have magical powers residing in its hair, but if the hair is removed or even grabbed, it may be rendered helpless.
</item 4>

<item 5>
Erotikon (1920 film)
Erotikon is a 1920 Swedish romantic comedy film directed by Mauritz Stiller, starring Tora Teje, Karin Molander, Anders de Wahl and Lars Hanson.

 It is based on the 1917 play "A kék róka" by Ferenc Herczeg.

 The story revolves around an entomology professor obsessed with the sexual life of bugs, and his easygoing wife who is courted by two suitors.
</item 5>

<item 6>
Vörðr
In Norse mythology, a vǫrðr (pl.

 varðir or verðir — "warden," "watcher" or "caretaker") is a warden spirit, believed to follow from birth to death the soul ("hugr") of every person.

 In Old Swedish, the corresponding word is "varþer"; in modern Swedish "vård", and the belief in them remained strong in Scandinavian folklore up until the last
centuries.

 The English word '"wraith" is derived from "vǫrðr", while "" and "warden" are cognates.
</item 6>

<item 7>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 7>

<item 8>
The Saga of Gosta Berling
The Saga of Gosta Berling (Swedish: Gösta Berlings saga ) is a 1924 Swedish romantic drama film directed by Mauritz Stiller and released by AB Svensk
Filmindustri, starring Lars Hanson, Gerda Lundequist and Greta Garbo in her native break-out role on film.

 The film is based on the 1891 debut novel of the same name by the Swedish author and Nobel Prize winner Selma Lagerlöf.

 It is also known as Gosta Berling's Saga, The Story of Gosta Berling and The Atonement of Gosta Berling.
</item 8>

<item 9>
Hulder
A hulder is a seductive forest creature found in Scandinavian folklore.

 (Her name derives from a root meaning "covered" or "secret".)

 In Norwegian folklore, she is known as huldra ("the [archetypal] hulder", though folklore presupposes that there is an entire Hulder race and not just a single
individual).

 She is known as the skogsrå "forest spirit" or Tallemaja "pine tree Mary" in Swedish folklore, and ulda in Sámi folklore.

 Her name suggests that she is originally the same being as the völva divine figure Huld and the German Holda.
</item 9>

<item 10>
Skogsrå
The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical
creature of the forest, or Rå, in Swedish folklore.
</item 10>
=== END ITEM id=ba78999987ec623f ===

=== ITEM id=a935be3491f3b4c5 ===
Gold sentences and facts:
Sentence 0:  The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
  Fact 0.0: The first of the books was optioned by Universal Pictures.
  Fact 0.1: The option followed a bidding war over the manuscript.
Sentence 1:  The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film, and the oldest in terms of the overall film market .
  Fact 1.0: The company was founded in 1912.
  Fact 1.1: The company was founded by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour.
  Fact 1.2: The company is the oldest surviving film studio in the United States.
  Fact 1.3: The company is the world's fourth oldest film studio, after Gaumont, Pathé and Nordisk Film.
  Fact 1.4: The company is the oldest film studio in terms of the overall film market.

Retrieved items:
<item 1>
Graph query row:
organization: Universal Pictures
founded_on: 1912
</item 1>

<item 2>
Joshua Bloom
Joshua Simon Bloom (born June 8, 1974 in Washington, D.C.) is an American astrophysicist, full professor of astronomy at the University of California, Berkeley,
and was the CTO and co-founder of the machine-learning company wise.io (acquired by General Electric, 2016).

 He received a Bachelor of Arts in astronomy and astrophysics and physics from the Harvard College in 1996, an M.Phil from Cambridge University in 1997, and a
PhD in astronomy from the California Institute of Technology in 2002.

 He was a Junior Fellow of the Harvard Society of Fellows from 2002 to 2005.

 His astronomy research focuses on gamma-ray bursts and other astrophysical transients such as supernovae and tidal disruption events.

 He is author of the book "What Are Gamma-Ray Bursts?"

 published by Princeton University Press in 2011.
</item 2>

<item 3>
Applewood Books
Applewood Books is a book publishing company founded by Phil Zuckerman in 1976.

 They specialize in publishing exacting recreations of historic books, including complex reprints of children's art and pop-up books and other books published
by methods which duplicate antique publishing techniques.

 They have more than 2000 titles in print.

 In recent years, the company has been working to increase the number of reprints it publishes.

 In 2007, the company published over 300 titles.
</item 3>

<item 4>
Mimeo, Inc
Mimeo.com, Inc. is a privately held Print on demand and digital distribution document company.

 It was the first to offer online printing and overnight delivery of complex documents and marketing materials.

 The company refers to itself as a technology company that prints.

 Customers utilize a proprietary online workflow connected to multiple print production, warehouse and distribution centers.

 Customers include small, mid-sized and large companies.

 The company was named after the Mimeograph.

 Printing and distribution centers are located in Memphis, Tennessee, Newark, New Jersey and Huntingdon, Cambridgeshire, England.
</item 4>

<item 5>
Field Enterprises
Field Enterprises was a private holding company founded on August 31, 1944, by Marshall Field III and others whose main asset was the "Chicago Sun".

 That same year the company acquired the book publishers Simon & Schuster and Pocket Books.
</item 5>

<item 6>
Bogle-L'Ouverture Publications
Bogle-L’Ouverture Publications (BLP) is a radical London-based publishing company founded in 1968 by Guyanese activists Jessica Huntley (23 February 1927 – 13
October 2013) and Eric Huntley (born 25 September 1929).

 Named in honour of two outstanding liberation fighters in Caribbean history, Toussaint L'Ouverture and Paul Bogle, the company began operating during a period
in the UK when "books by Black authors or written with a sympathetic view of Black people’s history and culture were rare in mainstream bookshops in the UK."

 Alongside New Beacon Books (founded in 1966) and Allison & Busby (founded in 1967), BLP was one of the first black-owned independent publishing companies in
the UK.

 BLP has been described as "a small, unorthodox, self-financing venture that brought a radical perspective to non-fiction, fiction, poetry and children's
books."
</item 6>

<item 7>
Simon Bloom
Simon Bloom is a series of young adult novels by Michael Reisman.

 Set in the fictional town of Lawnville, New Jersey, it is about a boy who finds a book that lets him control the laws of physics.

 The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
</item 7>

<item 8>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 8>

<item 9>
Jason Sizemore
Jason Sizemore is an American writer and editor based in Lexington, Kentucky.

 He is the owner and managing editor of Apex Publications.

 He was born in Big Creek, KY (pop.

 400).

 He was the editor and publisher of "Apex Digest", a quarterly science fiction and horror digest that ran for 12 issues between 2005 and 2008.

 As the publisher/managing editor of "Apex Magazine", he was nominated for the Hugo Award in the semiprozine category in 2012, 2013 and 2014.

 As a writer he has published several stories in genre magazines.

 His first short story collection, "Irredeemable", was published in April, 2014.
</item 9>

<item 10>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 10>
=== END ITEM id=a935be3491f3b4c5 ===

=== ITEM id=8e117360e84083f7 ===
Gold sentences and facts:
Sentence 0: Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.
  Fact 0.0: Innocence is a 2013 American horror drama film.
  Fact 0.1: Innocence was directed by Hilary Brougher.
  Fact 0.2: Hilary Brougher co-wrote Innocence with Tristine Skyler.
Sentence 1:  The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
  Fact 1.0: The movie Innocence stars Sophie Curtis.
  Fact 1.1: The movie Innocence stars Kelly Reilly.
  Fact 1.2: The movie Innocence stars Graham Phillips.
  Fact 1.3: The movie Innocence stars Linus Roache.
  Fact 1.4: The movie Innocence stars Sarah Sutherland.
  Fact 1.5: The movie Innocence stars Stephanie March.
Sentence 2: Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.
  Fact 2.0: Stephanie Caroline March was born on 23, 1974.
  Fact 2.1: Stephanie Caroline March is an American actress.
Sentence 3:  She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
  Fact 3.0: Stephanie March is best known for her role as Alexandra Cabot.
  Fact 3.1: The role of Alexandra Cabot was in a long-running NBC series.

Retrieved items:
<item 1>
Nancy Schreiber
Nancy Schreiber (born June 27, 1949) is an American cinematographer known for her work on "Chain of Desire", "Dead Beat", "The Celluloid Closet", "November",
and "The Nines".
</item 1>

<item 2>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 2>

<item 3>
In My Sleep
In My Sleep is a 2010 suspense thriller film written, directed and produced by film director Allen Wolf.

 It stars Philip Winchester, Lacey Chabert, Tim Draxl, Abigail Spencer and Kelly Overton.

 It also features Kirsten Vangsness from Criminal Minds in a cameo role and Tony Hale from Veep in a cameo as well.

 The story is about a massage therapist with chronic insomnia who fears he may have murdered a good friend while sleepwalking.
</item 3>

<item 4>
Stephanie March
Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.

 She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
</item 4>

<item 5>
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 5>

<item 6>
Gloria Garayua
Gloria Garayua (born October 18, 1978 in New York City) is an American film and television actress.

 Garayua made her major film debut in the 2005 comedy film "Fun With Dick and Jane", and is now commonly cast in guest roles on long-running series such as "Six
Feet Under", "Weeds" and "The Shield".

 After being cast in an ongoing role in "Grey's Anatomy", Garayua has played recurring roles in other popular TV shows such as "Cougar Town" and "How to Get
Away with Murder".
</item 6>

<item 7>
Kim Johnston Ulrich
Kim Johnston Ulrich (born March 24, 1955) is an American actress.

 From 1983 to 1986, Ulrich played the role of Diana McColl on "As the World Turns".

 In 1988, she guest-starred in two episodes of "Werewolf".

 In 1990, Ulrich appeared on the TV series "Wings" as Carol, Brian's ex-wife.

 In 1993, Ulrich appeared in the pilot episode of "" as Dr. Antoinette Baines, a villainous scientist.

 She appeared in a third season episode of "Highlander: The Series" in 1995.

 She appeared in the first episode of season 4 of "" as a conniving wife in "Murder by Friendly Fire" in 1996.
</item 7>

<item 8>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 8>

<item 9>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 9>

<item 10>
Conviction (2006 TV series)
Conviction is an American television drama on NBC that debuted as a mid-season replacement on Friday, March 3, 2006.

 The cast includes Stephanie March reprising her "" role as Alexandra Cabot.

 In the series, Cabot returns to New York City and becomes a Bureau Chief ADA supervising a group of young but talented assistant district attorneys after a
stint in the Witness Protection Program.

 Other cast members include Eric Balfour, Anson Mount, Jordan Bridges, Julianne Nicholson, Milena Govich, and J. August Richards.
</item 10>
=== END ITEM id=8e117360e84083f7 ===

=== ITEM id=6b11029484bb38fb ===
Gold sentences and facts:
Sentence 0: Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.
  Fact 0.0: Mirrors is a 2008 supernatural horror film.
  Fact 0.1: Mirrors was directed by Alexandre Aja.
  Fact 0.2: Kiefer Sutherland starred in Mirrors.
  Fact 0.3: Paula Patton starred in Mirrors.
  Fact 0.4: Amy Smart starred in Mirrors.
Sentence 1: Paula Maxine Patton (born December 5, 1975) is an American actress.
  Fact 1.0: Paula Maxine Patton was born on December 5, 1975.
  Fact 1.1: Paula Maxine Patton is an American actress.

Retrieved items:
<item 1>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 1>

<item 2>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 2>

<item 3>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 3>

<item 4>
Ela, Hela and The Hitch
Ela, Hela, and the Hitch is a play by Václav Havel.

 The play was written for the Artistic Director of the Theatre on the Balustrade, Ivan Vyskočil, as part of a longer evening, entitled "Hitchhiking".

 Along with "Ela, Hela, and the Hitch", Havel also wrote a sketch called "Motormorphosis".

 Reportedly, Vyskočil altered Havel’s sketches for the performance, though the original text was discovered by a Czech theater scholar, Lenka Jungmannová.

 "Motormorphosis", in a translation by Carol Rocamora, was performed at the Havel Festival in 2006, a world premiere of the text as written.

 "Ela, Hela, and the Hitch" premiered in an English translation by Edward Einhorn following a revival of "Motormorphosis" at New York’s Bohemian National Hall
in 2011.
</item 4>

<item 5>
Canadian actress Rachelle Lefevre starred in *Big Wolf on Campus*, played Victoria in the first two *Twilight* films, and appeared in *Off the Map*, *A Gifted
Man*, and *Under the Dome*. American actress Gloria Garayua debuted in *Fun with Dick and Jane* and had roles in *Grey’s Anatomy*, *Cougar Town*, and *How to
Get Away with Murder*. Zooey Deschanel’s films include *Almost Famous*, *All
</item 5>

<item 6>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 6>

<item 7>
Mirrors (film)
Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.

 The film was first titled "Into the Mirror", but the name was later changed to "Mirrors".

 Filming began on May 1, 2007, and it was released in American theaters on August 15, 2008.
</item 7>

<item 8>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
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
Hitch (film)
Hitch is a 2005 American romantic comedy film directed by Andy Tennant and starring Will Smith.

 The film, which was written by Kevin Bisch, co-stars Eva Mendes, Kevin James, and Amber Valletta.

 Smith plays the main fictional character of the film, Alex "Hitch" Hitchens, who is a professional dating consultant who makes a living teaching men how to woo
women.

 The film was released on February 11, 2005 by Columbia Pictures.
</item 10>
=== END ITEM id=6b11029484bb38fb ===

=== ITEM id=3a7c55afbf3e495e ===
Gold sentences and facts:
Sentence 0:  The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss' 1983 book "Fatal Vision".
  Fact 0.0: The journalist in question is Joe McGinniss.
  Fact 0.1: The murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald.
  Fact 0.2: Dr. Jeffrey R. MacDonald became the subject of Joe McGinniss' 1983 book 'Fatal Vision'.
Sentence 1: Jeffrey Robert MacDonald (born October 12, 1943) is an American medical doctor who was convicted in 1979 of murdering his pregnant wife and two daughters in February 1970.
  Fact 1.0: Jeffrey Robert MacDonald was born on October 12, 1943.
  Fact 1.1: Jeffrey Robert MacDonald is an American medical doctor.
  Fact 1.2: Jeffrey Robert MacDonald was convicted in 1979 of murdering his pregnant wife.
  Fact 1.3: Jeffrey Robert MacDonald was convicted in 1979 of murdering his two daughters.
  Fact 1.4: The murders took place in February 1970.

Retrieved items:
<item 1>
Joe McGinniss
Joseph R. McGinniss, Sr. (December 9, 1942 – March 10, 2014), known as Joe McGinniss, was an American non-fiction writer and novelist.

 The author of twelve books, he first came to prominence with the best-selling "The Selling of the President 1968" which described the marketing of
then-presidential candidate Richard Nixon.

 He is popularly known for his trilogy of bestselling true crime books — "Fatal Vision", "Blind Faith" and "Cruel Doubt" — which were adapted into TV miniseries
in the 1980s and 90s.

 His last book was "The Rogue: Searching for the Real Sarah Palin", an account of Sarah Palin, the former governor of Alaska who was the 2008 Republican
vice-presidential nominee.
</item 1>

<item 2>
Jeffrey R. MacDonald
Jeffrey Robert MacDonald (born October 12, 1943) is an American medical doctor who was convicted in 1979 of murdering his pregnant wife and two daughters in
February 1970.
</item 2>

<item 3>
Fatal Vision controversy
The controversy over Fatal Vision, journalist and author Joe McGinniss's best-selling 1983 true crime book, is a decades-long dispute spanning several court
cases and discussed in several other published works.
</item 3>

<item 4>
Fatal Vision (disambiguation)
Fatal Vision is the 1983 true crime book by Joe McGinniss which lies at the center of the "Fatal Vision" controversy.
</item 4>

<item 5>
Mazengarb Report
The Mazengarb Report of 1954, formally titled the "Report of the Special Committee on Moral Delinquency in Children and Adolescents", resulted from a
ministerial inquiry (the Special Committee on Moral Delinquency in Children and Adolescents) sparked primarily by two infamous and well-publicised events in New
Zealand's history: the 22 June 1954 Parker–Hulme murder case (subject of the 1994 Peter Jackson film "Heavenly Creatures") and the 20 June 1954 "Petone
incident".

 The report gained its name from the inquiry chairman, Queen's Counsel Ossie Mazengarb.
</item 5>

<item 6>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 6>

<item 7>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 7>

<item 8>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 8>

<item 9>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 9>

<item 10>
Fatal Vision (miniseries)
Fatal Vision is a 1984 American television miniseries based on the account, in the book of the same name, of the murders in 1970 at Fort Bragg of the wife and
daughters of U.S. Army officer Jeffrey R. MacDonald.
</item 10>
=== END ITEM id=3a7c55afbf3e495e ===
