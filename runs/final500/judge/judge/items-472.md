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

=== ITEM id=e2c6674d58db3a1d ===
Gold sentences and facts:
Sentence 0:  He is owner and chef of "Dear.
  Fact 0.0: He is the owner of Dear.
  Fact 0.1: He is the chef of Dear.
Sentence 1:  It airs on JTBC on Mondays at 21:30 (KST) beginning November 17, 2014.
  Fact 1.0: The show airs on JTBC.
  Fact 1.1: The show airs on Mondays at 21:30 (KST).
  Fact 1.2: The show begins airing on November 17, 2014.

Retrieved items:
<item 1>
Black (TV series)
Black () is an upcoming fantasy-thriller South Korean television series starring Song Seung-heon, Go Ara, Lee El and Kim Dong-jun.

 It is set to air on OCN starting October 14, 2017 on Saturdays and Sundays at 22:20 (KST) time slot, replacing "Save Me".
</item 1>

<item 2>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 2>

<item 3>
Please Take Care of My Refrigerator
Please Take Care of My Refrigerator () is a 2014 South Korean cooking-variety program starring by many chefs and celebrity guests.

 It airs on JTBC on Mondays at 21:30 (KST) beginning November 17, 2014.
</item 3>

<item 4>
Lee Won-il
Lee Won-il (; born December 12, 1979) is a South Korean chef and television personality.

 He is currently a cast member in the variety show "Please Take Care of My Refrigerator".

 He is owner and chef of "Dear.

 bread".
</item 4>

<item 5>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 5>

<item 6>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 6>

<item 7>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 7>

<item 8>
Baek Jong-won's Food Truck
Baek Jong-won's Food Truck () is a South Korean cooking-variety program which has been broadcast since August 28, 2015.

 The program is currently hosted by Baek Jong-won and Kim Sung-joo, it airs on SBS every Friday at 23:20 (KST).
</item 8>

<item 9>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
</item 9>

<item 10>
Cook Representative
Cook Representative (), also known as National Chef Team, is a 2016 South Korean cooking-variety program starring Kim Sung-joo, Ahn Jung-hwan, Kang Ho-dong,
Choi Hyun-seok, Sam Kim, Lee Won-il and, , is the spin-off of "Please Take Care of My Refrigerator".

 It aired on JTBC during Wednesdays at 22:50 (KST) beginning February 17, 2016.

 The series aired its last episode August 10, 2016 after concluding the finals match of the World Championship.
</item 10>
=== END ITEM id=e2c6674d58db3a1d ===

=== ITEM id=5b8bfc8e4c4806fc ===
Gold sentences and facts:
Sentence 0:  The film stars Juno Temple, Julia Garner, Alessandro Nivola, Maggie Siff, Olivia Luccardi and Philip Ettinger.
  Fact 0.0: The film stars Juno Temple.
  Fact 0.1: The film stars Julia Garner.
  Fact 0.2: The film stars Alessandro Nivola.
  Fact 0.3: The film stars Maggie Siff.
  Fact 0.4: The film stars Olivia Luccardi.
  Fact 0.5: The film stars Philip Ettinger.
Sentence 1:  She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight Rises".
  Fact 1.0: The actress has appeared in the film Black Mass.
  Fact 1.1: The actress has appeared in the film The Other Boleyn Girl.
  Fact 1.2: The actress has appeared in the film Wild Child.
  Fact 1.3: The actress has appeared in the film Atonement.
  Fact 1.4: The actress has appeared in the film Maleficent.
  Fact 1.5: The actress has appeared in the film The Three Musketeers.
  Fact 1.6: The actress has appeared in the film The Dark Knight Rises.

Retrieved items:
<item 1>
Cracks (film)
Cracks is a 2009 independent drama psychological thriller film starring Eva Green, James McGovern, Juno Temple, María Valverde, and Imogen Poots, which was
released theatrically in the UK and Ireland on 4 December 2009.

 In the United States it was released by IFC Films on 18 March 2011, and premiered on Showtime in late 2011.
</item 1>

<item 2>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 2>

<item 3>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 3>

<item 4>
Lovelace (film)
Lovelace is a 2013 American biographical drama film about porn actress Linda Boreman, better known as Linda Lovelace, star of "Deep Throat", a seminal 1972 film
at the forefront of the Golden Age of Porn.

 "Lovelace" covers her life from age 20 to 32.

 Directed by Rob Epstein and Jeffrey Friedman, the film was written by Andy Bellin and stars Amanda Seyfried, Peter Sarsgaard, Sharon Stone, Adam Brody, James
Franco, Chloë Sevigny, and Juno Temple.

 The film had its world premiere on January 22, 2013, at the 2013 Sundance Film Festival and opened in a U.S. limited release on August 9, 2013.
</item 4>

<item 5>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 5>

<item 6>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 6>

<item 7>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 7>

<item 8>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
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
Dirty Girl (2010 film)
Dirty Girl is a 2010 American coming of age comedy-drama film written and directed by Abe Sylvia.

 It stars Juno Temple, Milla Jovovich and William H. Macy.

 It premiered at the 2010 Toronto International Film Festival on September 12, 2010.

 It was distributed theatrically by The Weinstein Company on October 7, 2011.
</item 10>
=== END ITEM id=5b8bfc8e4c4806fc ===

=== ITEM id=6bb8afa32e59df9a ===
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
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
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
Allied invasion of Italy
The Allied invasion of Italy was the Allied amphibious landing on mainland Italy that took place on 3 September 1943 during the early stages of the Italian
Campaign of World War II.

 The operation was undertaken by General Sir Harold Alexander's 15th Army Group (comprising General Mark W. Clark and General Bernard Montgomery's British
Eighth Army) and followed the successful invasion of Sicily.

 The main invasion force landed around Salerno on 9 September on the western coast in Operation Avalanche, while two supporting operations took place in
Calabria (Operation Baytown) and Taranto (Operation Slapstick).
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
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
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
=== END ITEM id=6bb8afa32e59df9a ===

=== ITEM id=3b3bc20f01c5ca56 ===
Gold sentences and facts:
Sentence 0: Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
  Fact 0.0: Ciaran Gribbin was born in 1976.
  Fact 0.1: Ciaran Gribbin is a Northern Irish singer-songwriter.
  Fact 0.2: Ciaran Gribbin is a guitarist.
  Fact 0.3: Ciaran Gribbin is a producer.
  Fact 0.4: Ciaran Gribbin hails from Castledawson, Northern Ireland.
Sentence 1: Brett Allen Scallions (born December 21, 1971) is an American musician.
  Fact 1.0: Brett Allen Scallions was born on December 21, 1971.
  Fact 1.1: Brett Allen Scallions is an American musician.
Sentence 2:  He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
  Fact 2.0: Brett Allen Scallions is the lead vocalist of the post-grunge band Fuel.
  Fact 2.1: Brett Allen Scallions is the rhythm guitarist of the post-grunge band Fuel.
  Fact 2.2: Brett Allen Scallions is the lyricist of the post-grunge band Fuel.

Retrieved items:
<item 1>
Brett Scallions
Brett Allen Scallions (born December 21, 1971) is an American musician.

 He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
</item 1>

<item 2>
Puppet Strings
Puppet Strings is the fifth studio album by American rock band Fuel.

 Released March 4, 2014, it is their first studio effort since 2003's "Natural Selection" to feature original lead singer Brett Scallions and the first Fuel
album not to feature original songwriter/guitarist, Carl Bell and bassist Jeff Abercrombie.

 Additionally, no one who participated in the recording of Fuel's previous album "Angels & Devils" participated in the recording of "Puppet Strings".
</item 2>

<item 3>
Yeah! (Fuel song)
"Yeah!"

 is the first song released from Fuel's album "Puppet Strings".

 It is also the first song released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 3>

<item 4>
Graph query row:
person: Brett Scallions
relationship: MEMBER_OF
organization: Fuel
</item 4>

<item 5>
Circus Diablo
Circus Diablo is an American rock band, formed in early 2006 by Billy Morrison (vocals), Billy Duffy (lead guitar) and Ricky Warwick (rhythm guitar).

 Fuel frontman Brett Scallions and Velvet Revolver drummer Matt Sorum subsequently joined the band on bass and drums, respectively.

 To date, Circus Diablo have released one studio album, entitled "Circus Diablo".
</item 5>

<item 6>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 6>

<item 7>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 7>

<item 8>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 8>

<item 9>
Soul to Preach To
"Soul to Preach To" is the first single released from Fuel's album "Puppet Strings".

 It is also the first single released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 9>

<item 10>
The Best of Fuel
The Best of Fuel is a compilation album by American hard rock band Fuel.

 It contains songs from their first three full-length albums, "Sunburn", "Something Like Human", and "Natural Selection".

 All of the said albums featured vocalist Brett Scallions who departed from the group only a few months after the compilation's release.

 "The Best of Fuel" consists solely of the band's released singles, with the exception of "Million Miles" from "Natural Selection" (it features "Quarter" from
the same album instead).
</item 10>
=== END ITEM id=3b3bc20f01c5ca56 ===

=== ITEM id=a8f22e8fc8a97237 ===
Gold sentences and facts:
Sentence 0: MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen neighborhood of Manhattan, New York City.
  Fact 0.0: MiMA stands for 'Middle of Manhattan'.
  Fact 0.1: MiMA is a mixed-use building.
  Fact 0.2: MiMA is located at 450 West 42nd Street.
  Fact 0.3: MiMA is located between Dyer and 10th Avenues.
  Fact 0.4: MiMA is located in the Hell's Kitchen neighborhood of Manhattan, New York City.
Sentence 1:  Ground was broken in 2007 and topping out occurred in early August 2010.
  Fact 1.0: Ground was broken on MiMA in 2007.
  Fact 1.1: Topping out of MiMA occurred in early August 2010.
Sentence 2:  It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on floors 51 to 63, and a Yotel hotel on the lower levels.
  Fact 2.0: MiMA was designed by the Miami-based architecture firm of Arquitectonica.
  Fact 2.1: MiMA has 43 floors of luxury rentals on floors 7 to 50.
  Fact 2.2: MiMA has twelve floors of condominiums on floors 51 to 63.
  Fact 2.3: MiMA has a Yotel hotel on the lower levels.
Sentence 3:  The building's height is 204 meters (670 feet).
  Fact 3.0: The height of MiMA is 204 meters (670 feet).
Sentence 4: 270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan, New York City.
  Fact 4.0: 270 Park Avenue is also known as the JPMorgan Chase Tower.
  Fact 4.1: 270 Park Avenue was formerly known as the Union Carbide Building.
  Fact 4.2: 270 Park Avenue is a high-rise office building.
  Fact 4.3: 270 Park Avenue is located in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
Park Avenue House
The Park Avenue House is a high rise residential building located at 2305 Park Avenue in the Park Avenue Historic District in Downtown Detroit, Michigan.

 It was listed on the National Register of Historic Places in 1996.

 It should not be confused with the nearby Park Avenue Hotel, which was demolished in 2015.
</item 1>

<item 2>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 2>

<item 3>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 3>

<item 4>
MiMA (building)
MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen
neighborhood of Manhattan, New York City.

 Ground was broken in 2007 and topping out occurred in early August 2010.

 It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on
floors 51 to 63, and a Yotel hotel on the lower levels.

 The building's height is 204 meters (670 feet).
</item 4>

<item 5>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 5>

<item 6>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 6>

<item 7>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 7>

<item 8>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 8>

<item 9>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 9>

<item 10>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 10>
=== END ITEM id=a8f22e8fc8a97237 ===

=== ITEM id=f345f8a5f7faa5fe ===
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
Man or Astro-man?
Man or Astro-man?

 is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
</item 1>

<item 2>
Auburn University
Auburn University (AU or Auburn) is a public research university in Auburn, Alabama, United States.

 With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in
Alabama.

 Auburn University is one of the state's two public flagship universities.
</item 2>

<item 3>
1997 Tennessee Volunteers football team
The 1997 Tennessee Volunteers football team represented the University of Tennessee during the 1997 NCAA Division I-A football season.

 Quarterback Peyton Manning had already completed his degree in three years, and had been projected to be the top overall pick in the 1997 NFL Draft, but
returned to Tennessee for his senior year.

 The Volunteers opened the season with victories against Texas Tech and UCLA, but for the third time in his career, Manning fell to Florida, 33–20.

 The Vols won the rest of their regular season games, finishing 10–1, and advanced to the SEC Championship Game against Auburn.

 Down 20–7, Manning led the Vols to a 30–29 victory.

 Throwing for four touchdowns, he was named the game's MVP, but injured himself in the process.

 The #3 Vols were matched up with #2 Nebraska in the Orange Bowl.

 Had Tennessee won and top-ranked Michigan lost to Washington State in the Rose Bowl, the Vols would have been expected to win the national championship.
</item 3>

<item 4>
Manimal Vinyl
Manimal Vinyl is a Calabasas based record label founded in 2006 by experimental musician and fashion stylist, Paul Beahan.

 They were originally known as a psychedelic folk label that makes semi-annual tribute records for worldwide charities, having eventually switched to dance
music and progressive underground.

 They are the label who has release debut records from act ranging from Warpaint to Bat for Lashes and exclusive singles from Moby, Duran Duran, Carla Bruni,
Edward Sharpe and the Magnetic Zeros.

 In 2008, they founded the annual Manimal Festival in Pioneertown, California which ran from 2008-2010.

 The festival has featured line-ups with Edward Sharpe and the Magnetic Zeros, Warpaint, Ariel Pink and papercranes.

 Manimal has offices in Los Angeles and New York City.
</item 4>

<item 5>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 5>

<item 6>
No Talking, Just Head
No Talking, Just Head is an album released in 1996 by The Heads, a band composed of Jerry Harrison, Tina Weymouth, and Chris Frantz of Talking Heads, joined by
a variety of guest singers.

 Its name may be seen as an allusion to the fact that Talking Heads' former vocalist, David Byrne, is the only member not involved.
</item 6>

<item 7>
More Songs About Buildings and Food
More Songs About Buildings and Food is the second studio album by the American rock band Talking Heads, released in July 1978.

 The album was the first of three Talking Heads LPs produced by collaborator Brian Eno.

 It saw the group move musically toward a danceable style, crossing singer David Byrne's unusual delivery with new emphasis on the rhythm section (made up of
bassist Tina Weymouth and drummer Chris Frantz).
</item 7>

<item 8>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 8>

<item 9>
Talking Heads
Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.

 The band comprised David Byrne (lead vocals, guitar), Chris Frantz (drums), Tina Weymouth (bass), and Jerry Harrison (keyboards, guitar).

 Described by critic Stephen Thomas Erlewine as "one of the most critically acclaimed bands of the '80s," the group helped to pioneer new wave music by
integrating elements of punk, art rock, funk, and world music with avant-garde sensibilities and an anxious, clean-cut image.
</item 9>

<item 10>
Talking Heads Africa
Talking Heads (Africa) was introduced in Cape Town in 2008 as part of the Infecting the City public art festival.

 Talking Heads has four core components that form the project.

 These include: developing a platform for conversation and exchange with and between experts; creating a network of African thought leaders; shooting
mini-documentaries that define these leaders and their contributions; developing the tools to make this model work in cities all over the African continent.
</item 10>
=== END ITEM id=f345f8a5f7faa5fe ===

=== ITEM id=736ae4660c974def ===
Gold sentences and facts:
Sentence 0: Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.
  Fact 0.0: Neon City is also known as Anno 2053 in Italy.
  Fact 0.1: Neon City is also known as Neonski Grad in Serbia.
  Fact 0.2: Neon City is a 1991 Canadian film.
  Fact 0.3: Neon City is a post-apocalyptic science fiction film.
  Fact 0.4: Neon City was directed by Monte Markham.
  Fact 0.5: Neon City was written by Jeff Begun.
  Fact 0.6: Neon City was written by Ann Lewis Hamilton.
  Fact 0.7: Neon City's writing was credited under the pseudonym Buck Finch.
Sentence 1:  The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
  Fact 1.0: Michael Ironside is in the cast of Neon City.
  Fact 1.1: Vanity is in the cast of Neon City.
  Fact 1.2: Lyle Alzado is in the cast of Neon City.
  Fact 1.3: Valerie Wildman is in the cast of Neon City.
  Fact 1.4: Nick Klar is in the cast of Neon City.
  Fact 1.5: Juliet Landau is in the cast of Neon City.
  Fact 1.6: Richard Sanders is in the cast of Neon City.
  Fact 1.7: Monte Markham is in the cast of Neon City.
Sentence 2: Lyle Martin Alzado (April 3, 1949 – May 14, 1992) was a professional American football defensive end of the National Football League famous for his intense and intimidating style of play.
  Fact 2.0: Lyle Martin Alzado was born on April 3, 1949.
  Fact 2.1: Lyle Martin Alzado died on May 14, 1992.
  Fact 2.2: Lyle Martin Alzado was a professional American football defensive end.
  Fact 2.3: Lyle Martin Alzado played in the National Football League.
  Fact 2.4: Lyle Martin Alzado was famous for his intense and intimidating style of play.

Retrieved items:
<item 1>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 1>

<item 2>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 2>

<item 3>
Graph query row:
film: Neon City
titles: ['Anno 2053', 'Neonski Grad']
connected_person: None
relationship: None
role: None
</item 3>

<item 4>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 4>

<item 5>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 5>

<item 6>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 6>

<item 7>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 7>

<item 8>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 8>

<item 9>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 9>

<item 10>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 10>
=== END ITEM id=736ae4660c974def ===

=== ITEM id=64254c97025416b4 ===
Gold sentences and facts:
Sentence 0: Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".
  Fact 0.0: Stephanocereus is a genus of cactus.
  Fact 0.1: Stephanocereus is from Brazil.
  Fact 0.2: Stephanocereus is related to "Arrojadoa".
Sentence 1:  They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).
  Fact 1.0: The plants in this sentence are mostly liana-forming.
  Fact 1.1: The plants in this sentence are found in the Old World tropics.
  Fact 1.2: The plants in this sentence range from South China, Indonesia, Malaysia and the Philippines.
  Fact 1.3: The plants in this sentence range westward to Madagascar, with two species there.
  Fact 1.4: The plants in this sentence range westward to the Seychelles, with one species there.
  Fact 1.5: The plants in this sentence range southward to Australia, with three species there.
  Fact 1.6: The plants in this sentence range southward to New Caledonia, with one species there.
  Fact 1.7: The plants in this sentence range northward to India, with one species there.
  Fact 1.8: The plants in this sentence range northward to Sri Lanka, with one species there.

Retrieved items:
<item 1>
Nepenthes chaniana
Nepenthes chaniana ( ; after Datuk Chan Chew Lun, Managing Director of Natural History Publications (Borneo)) is a tropical pitcher plant species belonging to
the genus "Nepenthes".

 It is characterised by a dense indumentum of long, white hairs.

 Pitchers are cylindrical and mostly white to yellow in colouration.

 "Nepenthes chaniana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. boschiana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 1>

<item 2>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 2>

<item 3>
Nepenthes × kinabaluensis
Nepenthes × kinabaluensis , or the Kinabalu Pitcher-Plant, is the natural hybrid between "N. rajah" and "N. villosa".

 It was first collected near Kambarangoh on Mount Kinabalu, Borneo by Lilian Gibbs in 1910 and later mentioned by John Muirhead Macfarlane as ""Nepenthes" sp."

 in 1914.

 Although Macfarlane did not formally name the plant, he noted that "[a]ll available morphological details suggest that this is a hybrid between "N. villosa"
and "N. rajah"".

 It was finally described in 1976 by Shigeo Kurata as "N. × kinabaluensis".

 The name was first published in "Nepenthes of Mount Kinabalu", but was a "nomen nudum" at the time as it lacked an adequate description and information on the
type specimen.

 The name was subsequently published validly by Kurata in 1984.
</item 3>

<item 4>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 4>

<item 5>
Robert Cantley
Robert Cantley is a conservationist and Managing Director of Borneo Exotics, a Sri Lankan-based plant nursery specialising in tissue-cultured and seed-grown
"Nepenthes" species and hybrids.

 Cantley has contributed to a number of papers on "Nepenthes".

 The species "Nepenthes robcantleyi", the natural hybrid "Nepenthes × cantleyi" and the cultivar "Nepenthes" 'Cantley's Red' are named after him.
</item 5>

<item 6>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 6>

<item 7>
Nepenthes tenax
Nepenthes tenax ( ; from Latin: "tenax" "tenacious") is a lowland species of tropical pitcher plant native to northern Queensland, Australia.

 It is the third "Nepenthes" species recorded from the continent and its second endemic species.

 "Nepenthes tenax" is closely related to the two other Australian "Nepenthes" species: "N. mirabilis" and "N. rowaniae".
</item 7>

<item 8>
Nepenthes boschiana
Nepenthes boschiana ( ; after Johannes van den Bosch), or Bosch's Pitcher-Plant, is a tropical pitcher plant endemic to Borneo.

 It is most closely allied to "N. faizaliana".

 "Nepenthes borneensis" is considered a synonym of this species.

 "Nepenthes boschiana" has no known natural hybrids.

 No valid forms or varieties have been described.

 "Nepenthes boschiana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. chaniana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 8>

<item 9>
Nepenthes veitchii
Nepenthes veitchii ( or ; after James Veitch, nurseryman of the Veitch Nurseries), or Veitch's Pitcher-Plant, is a "Nepenthes" species from the island of
Borneo.

 The plant is widespread in north-western Borneo and can also be found in parts of Kalimantan.

 "Nepenthes veitchii" usually grows as an epiphyte, though the form from Bario seems to be strictly terrestrial and has not been observed to climb trees.
</item 9>

<item 10>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 10>
=== END ITEM id=64254c97025416b4 ===

=== ITEM id=1ada6e1abc16caa5 ===
Gold sentences and facts:
Sentence 0:  The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic Conference (MAAC).
  Fact 0.0: The Gaels were led by sixth year head coach Tim Cluess.
  Fact 0.1: The Gaels played their home games at the Hynes Athletic Center.
  Fact 0.2: The Gaels were members of the Metro Atlantic Athletic Conference (MAAC).
Sentence 1:  The Cyclones were coached by Steve Prohm, who was in his 1st season.
  Fact 1.0: The Cyclones were coached by Steve Prohm.
  Fact 1.1: Steve Prohm was in his 1st season.

Retrieved items:
<item 1>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 1>

<item 2>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 2>

<item 3>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 3>

<item 4>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 4>

<item 5>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 5>

<item 6>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 6>

<item 7>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 7>

<item 8>
Sylvester Croom
Sylvester Croom Jr. (born September 25, 1954) is an American football coach.

 He is currently the running backs coach for the NFL's Tennessee Titans.

 He was the head coach at Mississippi State University from 2004 to 2008, and the first African American head football coach in the Southeastern Conference.

 His father, Sylvester Croom, Sr., was himself an All-American football player at Alabama A&M, later the team chaplain at the University of Alabama, and has
been recognized by that school as one of the state's 40 pioneers of civil rights.

 Since his time at Mississippi State, Croom, Jr. has served as running backs coach for three teams in the National Football League.
</item 8>

<item 9>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 9>

<item 10>
Tim Bedore
Tim Bedore (born c. 1957) is an American comedian born in Chicago, IL.

 His parents moved to Stevens Point, Wisconsin when he was a child.

 He attended Pacelli High School (Wisconsin) in Stevens Point and the Appleton High School-West, where he graduated.
</item 10>
=== END ITEM id=1ada6e1abc16caa5 ===

=== ITEM id=0030731ffcce64f1 ===
Gold sentences and facts:
Sentence 0:  Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.
  Fact 0.0: Qionghai is called a city.
  Fact 0.1: Qionghai refers to a large land area in Hainan.
  Fact 0.2: The area of Qionghai in Hainan was once a county.
Sentence 1:  It has an area of 1692 km2 and in 2010, it had a population of 483,217.
  Fact 1.0: Qionghai has an area of 1692 km2.
  Fact 1.1: Qionghai had a population of 483,217 in 2010.
Sentence 2: Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.
  Fact 2.0: Suining has the Sichuanese Pinyin Xunin.
  Fact 2.1: Suining is a prefecture-level city.
  Fact 2.2: Suining is in eastern Sichuan province.
  Fact 2.3: Sichuan province is in Southwest China.
Sentence 3:  In 2002, Suining had a population of 658,798.
  Fact 3.0: Suining had a population of 658,798 in 2002.

Retrieved items:
<item 1>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 1>

<item 2>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 2>

<item 3>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 3>

<item 4>
Qionghai
Qionghai () is one of the seven county-level cities of Hainan province, China.

 Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.

 Within this area is the main city, Qionghai City.

 It is located in the east of the island at the mouth of the Wanquan River, 86 km from the provincial capital of Haikou.

 It has an area of 1692 km2 and in 2010, it had a population of 483,217.
</item 4>

<item 5>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 5>

<item 6>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 6>

<item 7>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 7>

<item 8>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 8>

<item 9>
Entries describe Chinese and Korean places, people, and organizations: Pungdo is an inhabited Yellow Sea island in Ansan, South Korea. Liaoning locations
include Pulandian (2,770 km²; 741,230 people), Longkou (620,000), Kaiyuan, Qinghe District, Shuangta, Taiyang, Xingcheng, and Kaiyuan West station. Politicians
Du Zheheng (1910–75), Liaoning’s first governor
</item 9>

<item 10>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 10>
=== END ITEM id=0030731ffcce64f1 ===

=== ITEM id=bcb3f53c6360be7b ===
Gold sentences and facts:
Sentence 0: San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San Antonio metropolitan area.
  Fact 0.0: San Antonio International Airport has the IATA code SAT.
  Fact 0.1: San Antonio International Airport has the ICAO code KSAT.
  Fact 0.2: San Antonio International Airport has the FAA LID SAT.
  Fact 0.3: San Antonio International Airport is an international airport.
  Fact 0.4: San Antonio International Airport is located in San Antonio, Texas.
  Fact 0.5: San Antonio International Airport serves the Greater San Antonio metropolitan area.
Sentence 1: Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.
  Fact 1.0: Yakutat Airport has the IATA code YAK.
  Fact 1.1: Yakutat Airport has the ICAO code PAYA.
  Fact 1.2: Yakutat Airport has the FAA LID YAK.
  Fact 1.3: Yakutat Airport is a state owned, public use airport.
  Fact 1.4: Yakutat Airport is located three nautical miles (6 km) southeast of the central business district of Yakutat.
  Fact 1.5: Yakutat is a city and borough in the U.S. state of Alaska.
  Fact 1.6: Yakutat has no road access to the outside world.

Retrieved items:
<item 1>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 1>

<item 2>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 2>

<item 3>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 3>

<item 4>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 4>

<item 5>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 5>

<item 6>
Stepanakert Airport
Stepanakert Airport (Armenian: Ստեփանակերտի Օդանավակայան ), is an airport in Khojaly, near Stepanakert, the capital city of the "de facto" independent Republic
of Artsakh (NKR) which is an internationally recognized part of Azerbaijan.

 The airport has been under the control of the Republic of Artsakh since 1992.

 Flights ceased to take place with the escalation of the Nagorno-Karabakh War in 1990.

 As the airport lies within the internationally recognized borders of Azerbaijan, and the current government is unrecognized, there are no codes for the airport
in the official IATA list.
</item 6>

<item 7>
Yakutat Airport
Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business
district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.

 Airline service is subsidized by the Essential Air Service program.
</item 7>

<item 8>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 8>

<item 9>
Pago Pago International Airport
Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of
the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the
United States.
</item 9>

<item 10>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 10>
=== END ITEM id=bcb3f53c6360be7b ===

=== ITEM id=4178a73d66b87947 ===
Gold sentences and facts:
Sentence 0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.
  Fact 0.0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate.
  Fact 0.1: Suzanne N.J. 'Susie' Chun Oakland represents the 13th District.
  Fact 0.2: Suzanne N.J. 'Susie' Chun Oakland has represented the 13th District since 1996.
Sentence 1:  Previously she was a member of the Hawaii House of Representatives from 1990-96.
  Fact 1.0: Suzanne N.J. 'Susie' Chun Oakland was previously a member of the Hawaii House of Representatives.
  Fact 1.1: Suzanne N.J. 'Susie' Chun Oakland was a member of the Hawaii House of Representatives from 1990 to 1996.
Sentence 2:  The lower chamber of the legislature is the Hawaiʻi House of Representatives.
  Fact 2.0: The lower chamber of the Hawaii legislature is the Hawaiʻi House of Representatives.

Retrieved items:
<item 1>
Oregon Territorial Legislature
Oregon’s Territorial Legislature was a bicameral legislative body created by the United States Congress in 1848 as the legislative branch of the government of
the Oregon Territory.

 The upper chamber Council and lower chamber House of Representatives first met in July 1849; they served as the region's legislative body until Oregon became a
state in February 1859, when they were replaced by the bicameral Oregon State Legislature.
</item 1>

<item 2>
Ron Cyrus
Ronald Ray "Ron" Cyrus (July 10, 1935 – February 28, 2006) was an American Democratic politician and public servant in Greenup County, Kentucky.

 He was the father of American country music singer/actor Billy Ray Cyrus and the paternal grandfather of Trace Cyrus, Miley Cyrus and Noah Cyrus.
</item 2>

<item 3>
Hawaii State Capitol
The Hawaii State Capitol is the official statehouse or capitol building of the U.S. state of Hawaii.

 From its chambers, the executive and legislative branches perform the duties involved in governing the state.

 The Hawaii State Legislature—composed of the twenty-five member Hawaii State Senate led by the President of the Senate and the fifty-one member Hawaii State
House of Representatives led by the Speaker of the House—convenes in the building.

 Its principal tenants are the Governor of Hawaii and Lieutenant Governor of Hawaii, as well as all legislative offices and the Legislative Reference Bureau.
</item 3>

<item 4>
Aloha Dalire
Fabianne Pomaialoha Wong Dalire, known professionally as Aloha Dalire, (June 22, 1950 – August 6, 2014) was an American Hawaiian kumu hula, or master hula
teacher.

 She won the first Miss Aloha Hula as Aloha Wong, in 1971, the same year that the Merrie Monarch Festival was established.

 The Miss Aloha Hula title is hula's top solo wahine (women's) honor.
</item 4>

<item 5>
Hawaii State Legislature
The Hawaii State Legislature is the state legislature of the U.S. state of Hawaii.

 The state legislature is a bicameral body consisting of a lower house, the Hawaii State House of Representatives, with 51 representatives, and an upper house,
the 25-member Hawaii State Senate.

 There are a total of 76 representatives in the legislature, each representing single member districts across the islands.

 The powers of the legislature are granted under Article III of the Constitution of Hawaii.
</item 5>

<item 6>
Suzanne Chun Oakland
Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.

 Previously she was a member of the Hawaii House of Representatives from 1990-96.

 She is generally considered to have liberal political views.
</item 6>

<item 7>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
</item 7>

<item 8>
Alaska Legislature
The Alaska Legislature is the state legislature of the U.S. state of Alaska.

 It is a bicameral institution, consisting of the lower Alaska House of Representatives, with 40 members, and the upper house Alaska Senate, with 20 members.

 There are 40 House Districts (1-40) and 20 Senate Districts (A-T).

 With a total of 60 lawmakers, the Alaska Legislature is the smallest bicameral state legislature in the United States and the second-smallest of all state
legislatures (only the 49-member unicameral Nebraska Legislature is smaller).

 There are no term limits for either chamber.
</item 8>

<item 9>
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
</item 9>

<item 10>
Hawaii Senate
The Hawaiʻi State Senate is the upper chamber of the Hawaii State Legislature.

 The senate consists of twenty-five members elected from an equal number of constituent districts across the islands.

 The senate is led by the President of the Senate, elected from the membership of the body, currently Ron Kouchi.

 The forerunner of the Hawaii State Senate during the government of the Kingdom of Hawaiʻ i was the House of Nobles originated in 1840.

 In 1894 the Constitution of the Republic of Hawaii renamed the upper house the present senate.

 Senators are elected to four-year terms and are not subject to term limits.

 Like most state legislatures in the United States, the Hawaii State Senate is a part-time body and senators often have active careers outside government.

 The lower chamber of the legislature is the Hawaiʻi House of Representatives.

 The membership of the Senate also elects additional officers to include the Senate Vice President, Senate Chief Clerk, Assistant Chief Clerk, Senate Sergeant
at Arms and Assistant Sergeant at Arms.
</item 10>
=== END ITEM id=4178a73d66b87947 ===
