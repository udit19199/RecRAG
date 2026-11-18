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

=== ITEM id=b0c41fb5eda2f522 ===
Gold sentences and facts:
Sentence 0: Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.
  Fact 0.0: Charles Thomas 'Tom' Johnston was born on August 15, 1948.
  Fact 0.1: Charles Thomas 'Tom' Johnston is an American musician.
  Fact 0.2: Charles Thomas Johnston is known by the name 'Tom'.
Sentence 1: Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.
  Fact 1.0: Daryl Palumbo was born on February 10, 1979.
  Fact 1.1: Daryl Palumbo is an American musician.
  Fact 1.2: Daryl Palumbo is originally from Elmira, New York.

Retrieved items:
<item 1>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 1>

<item 2>
Kim Johnston Ulrich
Kim Johnston Ulrich (born March 24, 1955) is an American actress.

 From 1983 to 1986, Ulrich played the role of Diana McColl on "As the World Turns".

 In 1988, she guest-starred in two episodes of "Werewolf".

 In 1990, Ulrich appeared on the TV series "Wings" as Carol, Brian's ex-wife.

 In 1993, Ulrich appeared in the pilot episode of "" as Dr. Antoinette Baines, a villainous scientist.

 She appeared in a third season episode of "Highlander: The Series" in 1995.

 She appeared in the first episode of season 4 of "" as a conniving wife in "Murder by Friendly Fire" in 1996.
</item 2>

<item 3>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 3>

<item 4>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 4>

<item 5>
Tom Johnston
Tom Johnston, Thomas Johnston or Tommy Johnston may refer to:
</item 5>

<item 6>
Richard T. Jones
Richard Timothy Jones (born January 16, 1972), known professionally as Richard T. Jones, is an American actor.

 Jones is best known for his portrayals of Laveinio in the dramatic film "The Wood" and Mike of the dramatic films "Why Did I Get Married?

" and "Why Did I Get Married Too?

".
</item 6>

<item 7>
Graph query row:
person: Daryl Palumbo
occupation: musician
</item 7>

<item 8>
Joseph Todaro Jr.
Joseph "Big Joe" Todaro Jr. (born 1945 or 1946) is a Buffalo, New York businessman and former organized crime figure involved in labor racketeering,
loansharking, illegal gambling, narcotics, and murder for hire.

 Joe Todaro Jr. became a business agent for the Laborers' International Union of North America (LIUNA) Local 210.

 In 1984, Joe Todaro Jr. allegedly became the underboss of the Buffalo crime family after his father became the new head of the crime family, upon the
retirement of his predecessor Samuel "Sam the Farmer" Frangiamore.

 In 1990, Joe Todaro Jr. resigned as business agent following investigations on the local's alleged ties to organized crime.

 Outside of organized crime, Todaro operates La Nova Pizzeria, a popular pizza restaurant in Buffalo.
</item 8>

<item 9>
Graph query row:
person: Tom Johnston
occupation: ['musician', 'guitarist', 'vocalist', 'founder', 'lead vocalist', 'songwriter']
</item 9>

<item 10>
Daryl Palumbo
Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.

 He is the frontman of the bands Glassjaw, Head Automatica and Color Film.

 As a youth he was a member of the Long Island straight edge band "XbustedX."

 In 1993 he met guitarist Justin Beck, who would become his good friend.

 Together they formed Glassjaw.
</item 10>
=== END ITEM id=b0c41fb5eda2f522 ===

=== ITEM id=74a5cdb10de7763c ===
Gold sentences and facts:
Sentence 0: Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.
  Fact 0.0: Disteghil Sar is also called Distaghil Sar.
  Fact 0.1: Disteghil Sar is the highest mountain in the Hispar Muztagh.
  Fact 0.2: The Hispar Muztagh is a subrange of the Karakoram mountain range.
  Fact 0.3: Disteghil Sar is in Gilgit-Baltistan.
Sentence 1:  The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
  Fact 1.0: Disteghil Sar has a top ridge about 3 km long.
  Fact 1.1: The top ridge of Disteghil Sar is above 7400 m.
  Fact 1.2: Disteghil Sar has three distinct summits.
  Fact 1.3: The (north)west summit of Disteghil Sar is 7885 m high.
  Fact 1.4: The central summit of Disteghil Sar is 7760 m high.
  Fact 1.5: The (south)east summit of Disteghil Sar is 7696 m or 7535 m high.
Sentence 2:  Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
  Fact 2.0: The summit of Disteghil Sar has an elevation of 7295 m.
  Fact 2.1: Disteghil Sar is the highest peak in the Yengisogat subrange of the Karakoram.

Retrieved items:
<item 1>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 1>

<item 2>
Passu Sar
Passu Sar (Urdu: ‎ ; or Passu Sar, Passu I) is a mountain peak in the Batura Muztagh, a sub-range of the Karakoram mountain range, located in the Gilgit
District of Gilgit-Baltistan, Pakistan, west of the Hunza Valley.

 It is the high point of the Passu massif, which also includes Passu Diar (or "Passu East", "Pasu II").

 The peak lies on the main ridge of the Batura Muztagh, about 7 km (4 mi) east of Batura Sar.
</item 2>

<item 3>
The entries cover Karakoram peaks and subranges, mostly in Gilgit-Baltistan, Pakistan, and Xinjiang, China. Hispar Muztagh’s highest peak is Distaghil Sar
(7,885 m), among the world’s highest; it has three summits. Other notable peaks include Kanjut Sar, Momhil Sar, Yutmaru Sar, Passu Sar, Mingli Sar (first
climbed by Nazir Sabir in 1988), Kez
</item 3>

<item 4>
Momhil Sar
Momhil Sar, or Mumhail Sar as pronounced in Wakhi is at 7343 m above sea level, is the 64th highest mountain peak in the world.

 Mumhail Sar in Wakhi means the mountain that overlooks or is above Grandmother's cattle pen or paddock.

 It is situated in the Hispar Muztagh subrange of the Karakoram range, a few kilometres to the north-west of its parent peak Trivor.
</item 4>

<item 5>
Kezhen Peak
Kezhen Peak, also known as Karpogo Sar, is a mountain in the Karakoram mountain range.

 It is located in the Xinjiang Autonomous Region of China.
</item 5>

<item 6>
Kanjut Sar
Kanjut Sar (Urdu: کنجت سر‎ ) or Kunjudh Sar as pronounced in "Wakhi" is a mountain located in the Hispar Muztagh, a subrange of the Karakoram mountain range.

 Kunjudh Sar in wakhi language mean that which overlooks Kunjudh, or above Kunjudh, while Khujudh is the wakhi name for Lower Hunza.

 It is the 26th highest mountain on Earth and the 11th highest in Pakistan.
</item 6>

<item 7>
The Crown (mountain)
The Crown, also known as Huang Guan Shan and sometimes Crown Peak, is a mountain in the Karakoram mountain range in China.

 It is located in the Xinjiang Uyghur Autonomous Region of China.

 Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
</item 7>

<item 8>
Yutmaru Sar
Yutmaru Sar is a mountain in the Hispar mountain range, a subrange of the Karakoram.

 At an elevation of 7283 m it is the 88th highest mountain in the world.

 Yutmaru Sar is located in Gilgit-Baltistan, Pakistan.

 It was first climbed in 1980.
</item 8>

<item 9>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 9>

<item 10>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 10>
=== END ITEM id=74a5cdb10de7763c ===

=== ITEM id=26a4edd619a30c1b ===
Gold sentences and facts:
Sentence 0:  He is best known for his roles in "How To Be Indie" (2009–2011), "Wingin' It" (2010–2013), "" (2012–2013), and "Open Heart" (2015).
  Fact 0.0: The person described is best known for a role in 'How To Be Indie' (2009–2011).
  Fact 0.1: The person described is best known for a role in 'Wingin' It' (2010–2013).
  Fact 0.2: The person described is best known for a role in an untitled work (2012–2013).
  Fact 0.3: The person described is best known for a role in 'Open Heart' (2015).
Sentence 1: Open Heart is a 2015 American-Canadian mystery-drama television series produced by the Epitome Pictures unit of DHX Media in association with marblemedia.
  Fact 1.0: Open Heart is a 2015 television series.
  Fact 1.1: Open Heart is an American-Canadian television series.
  Fact 1.2: Open Heart is a mystery-drama television series.
  Fact 1.3: Open Heart was produced by the Epitome Pictures unit of DHX Media.
  Fact 1.4: Open Heart was produced in association with marblemedia.

Retrieved items:
<item 1>
Studio 60’s 2006 pilot, written by Aaron Sorkin and directed by Thomas Schlamme, introduces a chaotic fictional sketch show. The Office’s “Two Weeks” (2009)
follows Michael’s planned paper-company startup and Pam’s copier struggle; its 2013 finale closes the series as episodes 200–201. Two and a Half Men ran from
2003 to 2015. Charlie Sheen’s final episode, “That Darn Priest,” preceded his
</item 1>

<item 2>
Wingin' It
Wingin' It (originally titled Angel on Campus prior to its debut) is a Canadian teen sitcom which aired on Family.

 The series was produced by Temple Street Productions in association with Family.

 It stars Demetrius Joyette and Dylan Everett.

 This show included guest-stars from other Family Channel shows such as "The Latest Buzz".

 It was announced that the series was renewed for a third season on June 13, 2011.

 Family announced that the series is not planned for a fourth season.
</item 2>

<item 3>
Dean Winchester
Dean Winchester is a fictional character and one of the two protagonists of the American drama television series "Supernatural".

 He is portrayed by Jensen Ackles.

 Other versions of the character having been portrayed by Hunter Brochu (toddler), Ridge Canipe (child), Nicolai Lawton-Giustra (pre-teen), Brock Kelly and
Dylan Everett (teen), and Chad Everett (elderly).
</item 3>

<item 4>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 4>

<item 5>
Open Heart (TV series)
Open Heart is a 2015 American-Canadian mystery-drama television series produced by the Epitome Pictures unit of DHX Media in association with marblemedia.

 It first aired on January 20, 2015, on TeenNick in the United States and previously on YTV; now currently on ABC Spark in Canada.
</item 5>

<item 6>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 6>

<item 7>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 7>

<item 8>
Dylan Everett
Dylan Everett (born January 24, 1995) is a Canadian actor.

 He is best known for his roles in "How To Be Indie" (2009–2011), "Wingin' It" (2010–2013), "" (2012–2013), and "Open Heart" (2015).
</item 8>

<item 9>
Dan Flannery
Arthur Daniel Flannery (born March 10, 1944) is an American actor from Evanston, Illinois.

 He has played roles in several movies, short films, and T.V. shows from a variety of genres including drama, action, and thriller.

 His filmography includes The Straight Story (1999), and Contagion (2011) and T.V. series such as Empire (2015 TV series) and Boss (2011-2012).
</item 9>

<item 10>
Everest (2015 film)
Everest is a 2015 British-American biographical adventure film directed and co-produced by Baltasar Kormákur, co-produced by Tim Bevan, Eric Fellner, Nicky
Kentish Barnes, Tyler Thompson and Brian Oliver and written by William Nicholson and Simon Beaufoy, adapted from Beck Weathers' memoir "Left for Dead: My
Journey Home from Everest" (2000).

 It stars an ensemble cast of Jason Clarke, Josh Brolin, John Hawkes, Robin Wright, Michael Kelly, Sam Worthington, Keira Knightley, Emily Watson, and Jake
Gyllenhaal.

 It is based on the real events of the 1996 Mount Everest disaster, and focuses on the survival attempts of two expedition groups, one led by Rob Hall (Clarke)
and the other by Scott Fischer (Gyllenhaal).
</item 10>
=== END ITEM id=26a4edd619a30c1b ===

=== ITEM id=71c390c49994c8f4 ===
Gold sentences and facts:
Sentence 0: On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.
  Fact 0.0: On February 25, 2017, this person lost to Jarrett Hurd.
  Fact 0.1: The February 25, 2017 loss was for the International Boxing Federation light middleweight world title.
Sentence 1: Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
  Fact 1.0: Jarrett Hurd was born on August 31, 1990.
  Fact 1.1: Jarrett Hurd is an American light middleweight boxer.
  Fact 1.2: Jarrett Hurd is from Accokeek, Maryland.

Retrieved items:
<item 1>
Andre Thysse
Andre Thysse ( (1968--) 07 1968 (age (2017)-(1968)-((11)<(02)or(11)==(02)and(30)<(07)) ) ) born in Germiston, is a South African professional super middle/light
heavyweight boxer of the 1990s and 2000s who won the Gauteng super middleweight Title, South African super middleweight title, and Commonwealth super
middleweight title, and was a challenger for the International Boxing Organization (IBO) super middleweight title against Brian Magee, World Boxing Council
(WBC) super middleweight title against Markus Beyer, World Boxing Council (WBC) International super middleweight title against Mikkel Kessler, WBC International
super middleweight title against Jürgen Brähmer, World Boxing Council (WBC) Continental Americas super middleweight title against Lucian Bute, World Boxing
Council (WBC) International light heavyweight title against Adrian Diaconu, and International Boxing Federation (IBF) Australasian super middleweight title
against Sakio Bika, his professional fighting weight varied from 163+3/4 lb , i.e.
</item 1>

<item 2>
Steve Foster (boxer)
Steve "The Viking" Foster ( (1960--) 28 1960 (age 56 ) ) born in Salford is a retired English professional boxer of the 1980s and '90s who won the Commonwealth
light middleweight title and IBF Inter-Continental light middleweight title, and was a challenger for the World Boxing Association (WBA) Inter-Continental light
middleweight title against Shaun Cummins, BBBofC British light middleweight title against Robert McCracken, International Boxing Federation (IBF)
Inter-Continental light middleweight title against Bahre Ahmeti, World Boxing Organization (WBO) light middleweight title against Ronald "Winky" Wright, BBBofC
British middleweight title against Howard Eastman, World Boxing Federation (WBF) middleweight title against Cornelius Carr, and International Boxing
Organization (IBO) middleweight title against Mpush Makambi, his professional fighting weight varied from 146 lb , i.e. welterweight to 167 lb , i.e. super
middleweight.
</item 2>

<item 3>
Kevin Kelly (boxer)
Kevin "Bones" Kelly (born August 7, 1969 in La Perouse, New South Wales) is an Australian retired professional welter/light middle/middleweight boxer of the
1990s and 2000s who won the South Australia State welterweight title, New South Wales (Australia) State welterweight title, New South Wales (Australia) State
light middleweight title, Australian light middleweight title, Pan Asian Boxing Association (PABA) light middleweight title, World Boxing Union (WBU)
middleweight title, and Commonwealth light middleweight title (twice), and was a challenger for the World Boxing Association (WBA) World light middleweight
title against David Reid, and World Boxing Organization (WBO) Inter-Continental light middleweight title against Gary Lockett, his professional fighting weight
varied from 142 lb , i.e. welterweight to 158+1/4 lb , i.e. middleweight.
</item 3>

<item 4>
Guy Waters
Guy "Arc Angel" Waters ( (1964--) 25 1964 (age (2017)-(1964)-((11)<(01)or(11)==(01)and(30)<(25)) ) ) is an Australian professional welter/light
middle/middle/super middle/light heavy/cruiserweight boxer of the 1980s, 1990s and 2000s who won the New South Wales State (Australia) light heavyweight title,
Australian light heavyweight title, Oriental and Pacific Boxing Federation (OPBF) light heavyweight title, Australasian Light Heavyweight Title, World Boxing
Federation (WBF) light heavyweight title, International Boxing Federation (IBF) Pan Pacific super middleweight title, Oriental and Pacific Boxing Federation
(OPBF) super middleweight title, and Commonwealth light heavyweight title, and was a challenger for the WBC light heavyweight title against Dennis Andries,
World Boxing Association (WBA) World light heavyweight title against Virgil Hill, World Boxing Council (WBC) cruiserweight title against Juan Carlos Gómez, and
Commonwealth super middleweight title against David Starie, his professional fighting weight varied from 167+1/2 lb , i.e.
</item 4>

<item 5>
Tony Harrison (boxer)
On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.

 The fight was originally supposed to be a title eliminator for the right to fight Jermall Charlo, but when Charlo vacated the title, the fight was elevated to
be for the vacant title.

 Harrison boxed well for majority of the first half of the fight but began to fade thereafter the finish came in round 9 when Hurd landed a right hand that
floored Harrison,
</item 5>

<item 6>
Scott Dixon (boxer)
featherweight to 167+1/2 lb , i.e. Super middleweight.
</item 6>

<item 7>
Jarrett Hurd
Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
</item 7>

<item 8>
Richard Williams (boxer)
Richard "The Secret" Williams ( (1971--) 09 1971 (age (2017)-(1971)-((11)<(05)or(11)==(05)and(30)<(09)) ) is an English professional light middle/middle/super
middleweight boxer of the 1990s, 2000s and 2010s who won the International Boxing Organization (IBO) light middleweight title, and Commonwealth light
middleweight title (twice), and was a challenger for the International Boxing Organization (IBO) light middleweight title against Sergio Gabriel Martinez, and
British Boxing Board of Control (BBBofC) British middleweight title against Howard Eastman, his professional fighting weight varied from 149+3/4 lb , i.e. light
middleweight to 162+1/4 lb , i.e. super middleweight.

 Richard Williams is trained by Brian Lawrence, and managed by Barry Hearn, and John Rooney.
</item 8>

<item 9>
Nedal Hussein
Nedal "Skinny" Hussein (born 1 December 1977) is an Australian professional bantam/super bantam/feather/super feather/Lightweight boxer of the 1990s and 2000s
who won the Australian super bantamweight title, Australian bantamweight title, International Boxing Federation (IBF) Pan Pacific featherweight title, World
Boxing Federation (WBF) featherweight title, World Boxing Union (WBU) super bantamweight title, World Boxing Organization (WBO) Asia Pacific super featherweight
title, International Boxing Federation (IBF) Pan Pacific super featherweight title, International Boxing Organization (IBO) Inter-Continental super
featherweight title, and Commonwealth super bantamweight title, and was a challenger for the World Boxing Council (WBC) International super bantamweight title
against Manny Pacquiao, World Boxing Council (WBC) super bantamweight title against Óscar Larios, World Boxing Organization (WBO) featherweight title against
Scott Harrison, Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki,
</item 9>

<item 10>
Scott Dixon (boxer)
"Super" Scott Dixon ( (1976--) 28 1976 (age (2017)-(1976)-((11)<(09)or(11)==(09)and(30)<(28)) ) ) born in Hamilton is a Scottish professional feather/super
feather/light/light welter/welter/light middle/middle/super middleweight boxer of the 1990s, 2000s and 2010s, who won the British Boxing Board of Control
(BBBofC) Scottish Area welterweight title, World Boxing Board (WBB) welterweight title, World Boxing Union (German Version) super middleweight title, and
Commonwealth welterweight title, and was a challenger for the World Athletic Association (WAA) welterweight title against Michael Carruth, BBBofC British
welterweight title against Derek Roche, World Boxing Federation (WBF) light middleweight title against Steve Roberts, World Boxing Organization (WBO)
Inter-Continental light middleweight title against Anthony Farnell, and World Boxing Union (WBU) light middleweight title against Mehrdad Takalobigashi, his
professional fighting weight varied from 125 lb , i.e.
</item 10>
=== END ITEM id=71c390c49994c8f4 ===

=== ITEM id=5fc2db26ecc09089 ===
Gold sentences and facts:
Sentence 0:  It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
  Fact 0.0: The described act consists of a duet.
  Fact 0.1: Dave Hemingway is a vocalist featured in the duet.
  Fact 0.2: Briana Corrigan is a vocalist featured in the duet.
Sentence 1: Briana Corrigan (born 30 May 1965) is a Northern Irish singer.
  Fact 1.0: Briana Corrigan was born on 30 May 1965.
  Fact 1.1: Briana Corrigan is a Northern Irish singer.

Retrieved items:
<item 1>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 1>

<item 2>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 2>

<item 3>
Takes a Little Time (Amy Grant song)
"Takes a Little Time" was a maxi-single released in 1997 (see 1997 in music) to promote Amy Grant's album "Behind the Eyes", which was also released that year.

 "Takes a Little Time" included two songs from "Behind the Eyes", as well as a new version of Grant's 1982 Christian radio hit, "El Shaddai".

 The maxi-single was also an enhanced CD (ECD), meaning it had a video that consumers could watch when they put it in their home computer.

 The ECD portion of the CD contained a live acoustic version of the song After the Fire, which Grant later released on her 2003 album "Simple Things"; Grant
stated on Oprah that she had written the song for her mother.
</item 3>

<item 4>
The Beautiful South
The Beautiful South was an English pop/rock group formed in 1988 by two former members of the Hull group the Housemartins—Paul Heaton and Dave Hemingway—both of
whom performed lead and backing vocals.

 Other members throughout the band's tenure were former Housemartins roadie Sean Welch (bass), Dave Stead (drums) and Dave Rotheray (guitar).

 After the band's first album (recorded as a quintet), they were joined by a succession of female vocalists, all of whom performed lead and backing vocals
alongside Heaton and Hemingway – Briana Corrigan for albums two and three after appearing as a guest vocalist on one, followed by Jacqui Abbott for the fourth
through seventh albums, and finally Alison Wheeler for the final three Beautiful South albums.
</item 4>

<item 5>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 5>

<item 6>
Briana Corrigan
Briana Corrigan (born 30 May 1965) is a Northern Irish singer.

 She was the first female singer for The Beautiful South, from 1988 to 1992.
</item 6>

<item 7>
Just a Little Time
Just a Little Time is an album by Lynsey de Paul.

 It was originally released in 30 September 1994 on the Music Deluxe (catalogue number MSCD9) label but has since been released on the Tring International PLC
label (catalogue number JHD128) and on Arc Records/The Magic Collection (Catalogue number MEC 949080).

 All of the songs on the album are written or co-written by de Paul.

 The songs "Sugar Me", Getting a Drag", "Storm in a Teacup and "Dancing on a Saturday Night" are radical re-recordings of her earlier hits performed in an
updated and uptempo fashion.

 "Won't Somebody Dance with Me" is still performed as a ballad, but has been updated and extended with an extra refrain.

 Never before released songs on the album include the ballads "Words Don't Mean A Thing" and title track "Just a Little Time" as well as the more uptempo
"Instant Love" and "Now & Then".
</item 7>

<item 8>
Hugh Whitaker
Hugh Whitaker (born 18 May 1961) is an English musician and the former drummer for the British indie rock band The Housemartins.

 He replaced original drummer Chris Lang and drummed for the band's first album, "London 0 Hull 4", and its attendant single releases.

 He left the band before the recording of their second album, "The People Who Grinned Themselves to Death".

 Whitaker left the band on amicable terms and even participated in the promotional video for the band's first single without him, "Five Get Over Excited",
wherein he was kidnapped by his replacement, Dave Hemingway, and locked in a hessian sack.
</item 8>

<item 9>
Takes a Little Time (Total Contrast song)
"Takes a Little Time" is a 1985 dance hit by British, techno-soul, duo, Total Contrast.

 Their debut American release went to number one on the U.S. dance chart for one week.

 "Takes a Little Time" did not make the Hot 100 but, peaked at number eighty on the R&B singles chart.
</item 9>

<item 10>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 10>
=== END ITEM id=5fc2db26ecc09089 ===

=== ITEM id=7d9ee6839fad957a ===
Gold sentences and facts:
Sentence 0:  It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke Bader and Linda Riley.
  Fact 0.0: The publication was launched in March 1994.
  Fact 0.1: The publication was launched by Millivres Ltd.
  Fact 0.2: The publication was launched under the editorship of Frances Williams.
  Fact 0.3: The publication is now published by Twin Media Group.
  Fact 0.4: Twin Media Group is owned by Silke Bader.
  Fact 0.5: Twin Media Group is owned by Linda Riley.
Sentence 1: Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.
  Fact 1.0: Elle is a worldwide lifestyle magazine.
  Fact 1.1: Elle is of French origin.
  Fact 1.2: Elle focuses on fashion.
  Fact 1.3: Elle focuses on beauty.
  Fact 1.4: Elle focuses on health.
  Fact 1.5: Elle focuses on entertainment.
Sentence 2:  "Elle" is also the world's best-selling fashion magazine.
  Fact 2.0: Elle is also the world's best-selling fashion magazine.
Sentence 3:  It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.
  Fact 3.0: Elle was founded by Pierre Lazareff.
  Fact 3.1: Elle was founded by Hélène Gordon, the wife of Pierre Lazareff.
  Fact 3.2: Elle was founded in 1945.

Retrieved items:
<item 1>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 1>

<item 2>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 2>

<item 3>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 3>

<item 4>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
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
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 6>

<item 7>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 7>

<item 8>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 8>

<item 9>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 9>

<item 10>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 10>
=== END ITEM id=7d9ee6839fad957a ===

=== ITEM id=4b37e0035ad5471b ===
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
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 1>

<item 2>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 2>

<item 3>
Cantor–Bernstein theorem
In set theory and order theory, the Cantor–Bernstein theorem states that the cardinality of the second type class, the class of countable order types, equals
the cardinality of the continuum.

 It was used by Felix Hausdorff and named by him after Georg Cantor and Felix Bernstein.

 Cantor constructed a family of countable order types with the cardinality of the continuum, and in his 1901 inaugural dissertation Bernstein proved that such a
family can have no higher cardinality.
</item 3>

<item 4>
Georg Cantor
Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.

 He invented set theory, which has become a fundamental theory in mathematics.

 Cantor established the importance of one-to-one correspondence between the members of two sets, defined infinite and well-ordered sets, and proved that the
real numbers are more numerous than the natural numbers.

 In fact, Cantor's method of proof of this theorem implies the existence of an "infinity of infinities".

 He defined the cardinal and ordinal numbers and their arithmetic.

 Cantor's work is of great philosophical interest, a fact of which he was well aware.
</item 4>

<item 5>
Alfred von Arneth
Alfred Ritter von Arneth (10 July 181930 July 1897) was an Austrian historian.

 His principal scholarly work is a ten-volume biography of the Habsburg Empress Maria Theresa, first published in installments from 1863 to 1879 and still
regarded as the standard work on the subject.

 Born at Vienna, he was the son of Joseph Calasanza von Arneth (1791–1863), a well-known historian and archaeologist, who wrote a history of the Austrian Empire
(Vienna, 1827) and several works on numismatics and brother of Doctor Franz Hektor von Arneth (1818–1907).
</item 5>

<item 6>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 6>

<item 7>
Cantor space
In mathematics, a Cantor space, named for Georg Cantor, is a topological abstraction of the classical Cantor set: a topological space is a Cantor space if it is
homeomorphic to the Cantor set.

 In set theory, the topological space 2 is called "the" Cantor space.

 Note that, commonly, 2 is referred to simply as the Cantor set, while the term Cantor space is reserved for the more general construction of "D" for a finite
set "D" and a set "S" which might be finite, countable or possibly uncountable.
</item 7>

<item 8>
The entries cover Ravenel’s stable-homotopy conjectures (largely proved except the open telescope conjecture); linguistic binomial pairs; mathematician Hermann
Brunn and Brunnian links; Jónsson–Tarski algebras; and Cantor-related topics: the Cantor–Bernstein theorem, proper classes, Cantor space, the symbol Tav for an
inconsistent collection of cardinals, Cantor’s mathematics and first set-theory paper, the fat Cantor
</item 8>

<item 9>
Jónsson–Tarski algebra
In mathematics, a Jónsson–Tarski algebra or Cantor algebra is an algebraic structure encoding a bijection from an infinite set "X" onto the product "X"×"X".

 They were introduced by .

 , named them after Georg Cantor because of Cantor's pairing function and Cantor's theorem that an infinite set "X" has the same number of elements as "X"×"X";
the term "Cantor algebra" is also occasionally used to mean the Boolean algebra of all clopen subsets of the Cantor set, or the Boolean algebra of Borel subsets
of the reals modulo meager sets (sometimes called the Cohen algebra).
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
=== END ITEM id=4b37e0035ad5471b ===

=== ITEM id=17bdc0108389dc85 ===
Gold sentences and facts:
Sentence 0: In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
  Fact 0.0: The Fader published an interview with CupcakKe in September.
  Fact 0.1: In the interview, CupcakKe discussed her new album.
  Fact 0.2: CupcakKe's new album would address "rape, and everything else".
Sentence 1: The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.
  Fact 1.0: The Fader, stylized as The FADER, is a New York City-based music magazine.
  Fact 1.1: The Fader was launched in 1999.
  Fact 1.2: The Fader was launched by Rob Stone and Jon Cohen.
  Fact 1.3: The Fader covers music, style and culture.
Sentence 2:  It was the first print publication to be released on iTunes.
  Fact 2.0: The Fader was the first print publication to be released on iTunes.

Retrieved items:
<item 1>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 1>

<item 2>
Audacious (CupcakKe album)
In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
</item 2>

<item 3>
Elysia Crampton Presents: Demon City
Elysia Crampton Presents: Demon City is the second studio album by American electronic musician Elysia Crampton.

 It consists of a series of collaborations with fellow underground and queer electronic artists and has been described as an "epic poem" by its creator.

 It was released on July 22, 2016, on Break World Records.
</item 3>

<item 4>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 4>

<item 5>
Cornerstone (agency)
Cornerstone is a New York-based creative lifestyle marketing and public relations agency with offices in New York City, Los Angeles, and London.

 It is helmed by co-CEOs Rob Stone and Jon Cohen.

 The company employs approximately 100 professionals worldwide, in addition to a network of field marketing representatives throughout the United States.
</item 5>

<item 6>
State (magazine)
State.ie (formerly State Magazine) is an Irish website and formerly a monthly music magazine, which launched in April 2008 and ceased to print in January 2009
having published a total of ten issues.

 The magazine continues online and was voted Best Music Site in 2008 and Best Web Publication in 2010 in the Irish Web Awards.

 Originally the concept of the magazine involved a hard copy of which there was a charge to purchase, however after six issues it was decided to make the
magazine's content free both online and in print.

 The first issue, April 2008, appeared on Irish shelves on 6 March 2008 and featured Michael Stipe of R.E.M. on the cover.

 This immediately garnered comparisons between the new magazine's similarities with "Hot Press" who featured Stipe on their cover at the same time, a move
widely thought to be an attempt by Hot Press to stifle State's status as a serious 'alternative' to the more established local magazine.

 At a price of €5.50, "State" charged €2 more than "Hot Press".
</item 6>

<item 7>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
</item 7>

<item 8>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 8>

<item 9>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 9>

<item 10>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 10>
=== END ITEM id=17bdc0108389dc85 ===

=== ITEM id=dc178db02dba2f59 ===
Gold sentences and facts:
Sentence 0: Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.
  Fact 0.0: Andrew and Jeremy Get Married is a 2004 film.
  Fact 0.1: Andrew and Jeremy Get Married is a British documentary film.
  Fact 0.2: Andrew and Jeremy Get Married was written by Don Boyd.
  Fact 0.3: Andrew and Jeremy Get Married was directed by Don Boyd.
  Fact 0.4: Andrew and Jeremy Get Married was made for the BBC.
Sentence 1: The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.
  Fact 1.0: The Last Lions is a 2011 film.
  Fact 1.1: The Last Lions is an African nature documentary film.
  Fact 1.2: The Last Lions is by the National Geographic Society.
  Fact 1.3: The Last Lions was videotaped by Dereck and Beverly Joubert.
  Fact 1.4: The Last Lions was directed by Dereck and Beverly Joubert.

Retrieved items:
<item 1>
Tyler Perry's For Better or Worse
Tyler Perry's For Better or Worse is an American television sitcom created, written and executive produced by Tyler Perry.

 The series is based on and is a TV spin-off to Perry's 2007 film "Why Did I Get Married?

" and its 2010 sequel "Why Did I Get Married Too?

".

 The show premiered on TBS on November 25, 2011 and ended on July 22, 2017.

 Led by the comical, over-the-top antics of Marcus and Angela Williams, the ensemble follows three couples: Marcus and Angela, Joseph and Leslie, and Richard
and Keisha who are at various stages of their relationships as they go through the ups-and-downs of married life and dating.

 The third season of the series premiered on September 18, 2013, on the Oprah Winfrey Network.
</item 1>

<item 2>
Richard T. Jones
Richard Timothy Jones (born January 16, 1972), known professionally as Richard T. Jones, is an American actor.

 Jones is best known for his portrayals of Laveinio in the dramatic film "The Wood" and Mike of the dramatic films "Why Did I Get Married?

" and "Why Did I Get Married Too?

".
</item 2>

<item 3>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 3>

<item 4>
Andrew and Jeremy Get Married
Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.

 It tells the story of two Englishmen, Andrew Thomas and Jeremy Trafford, as they plan for their commitment ceremony.

 Originally commissioned for the "BBC Storyville" series, the film premiered at the 2004 Toronto International Film Festival.
</item 4>

<item 5>
Graph query row:
film: Andrew and Jeremy Get Married
type: documentary film
genre: None
relationship: None
connected_genre: None
</item 5>

<item 6>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 6>

<item 7>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 7>

<item 8>
Get Married 3
Get Married 3 is an Indonesian romantic comedy directed by Monty Tiwa and released in 2011.

 A sequel to Hanung Bramantyo's "Get Married" and "Get Married 2", it stars Nirina Zubir and Fedi Nuril as a married couple attempting to raise their triplets
while under intense pressure from their family and friends.

 The film was a commercial success and received favourable reviews in "The Jakarta Post" and "Suara Karya".
</item 8>

<item 9>
Graph query row:
film: The Last Lions
type: 2011 African nature documentary film
genre: None
relationship: None
connected_genre: None
</item 9>

<item 10>
Two Brides and a Baby
Two Brides and a Baby is a 2011 Nigerian romantic drama film directed by Teco Benson, starring Keira Hewatch, Kalu Ikeagwu, OC Ukeje, Chelsea Eze, Stella
Damasus-Aboderin and Okey Uzoeshi.

 It premiered on November 17, 2011.

 It received awards and nominations at Africa Movie Academy Awards, Best of Nollywood Awards and Africa Magic Viewers Choice Awards.
</item 10>
=== END ITEM id=dc178db02dba2f59 ===

=== ITEM id=e2e8fcac445b2204 ===
Gold sentences and facts:
Sentence 0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area women role models over 85 years old who remained artistically active.
  Fact 0.0: Still Kicking: Six Artistic Women of Project Arts & Longevity is a documentary film.
  Fact 0.1: Still Kicking: Six Artistic Women of Project Arts & Longevity was released in 2006.
  Fact 0.2: Still Kicking: Six Artistic Women of Project Arts & Longevity runs 32 minutes.
  Fact 0.3: Still Kicking: Six Artistic Women of Project Arts & Longevity was made by Pacific Grove filmmaker Greg Young.
  Fact 0.4: Still Kicking: Six Artistic Women of Project Arts & Longevity features six Bay Area women role models.
  Fact 0.5: The six women featured in Still Kicking are over 85 years old.
  Fact 0.6: The six women featured in Still Kicking remained artistically active.
Sentence 1: Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
  Fact 1.0: Time Bombs is a film.
  Fact 1.1: Time Bombs is a 2008 film.
  Fact 1.2: Time Bombs is a Canadian film.
  Fact 1.3: Time Bombs was produced by Productions de la ruelle.

Retrieved items:
<item 1>
Still Kicking: The Fabulous Palm Springs Follies
Still Kicking: The Fabulous Palm Springs Follies is a 1997 American short documentary film directed by Mel Damski.

 It was nominated at the 70th Academy Awards for Best Documentary Short Subject.

 It features The Fabulous Palm Springs Follies, which was formerly staged at the Palm Springs, California Plaza Theatre.
</item 1>

<item 2>
Time Bombs
Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
</item 2>

<item 3>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 3>

<item 4>
The Time to Live and the Time to Die
The Time to Live and the Time to Die, also known as A Time to Live, A Time to Die is a 1985 film directed by Hou Hsiao-hsien.

 This film is inspired by screenwriter-turned-director Hou's own coming-of-age story.
</item 4>

<item 5>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 5>

<item 6>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 6>

<item 7>
Time Bomb High School
Time Bomb High School is the second album by the Reigning Sound.

 It was released in 2002 on In the Red Records.

 The album featured the original Reigning Sound lineup of Greg Cartwright on lead vocals and guitar; Alex Greene on organ, piano, guitar, and backing vocals;
Jeremy Scott on bass, and backing vocals; and Greg Roberson on drums.

  Howlin' Pelle Almqvist of The Hives commented in "Rolling Stone" that Time Bomb High School was his favorite record of 2002.
</item 7>

<item 8>
The Short-Timers
The Short-Timers is a 1979 semi-autobiographical novel by U.S. Marine Corps veteran Gustav Hasford, about his experience in the Vietnam War.

 It was later adapted into the film "Full Metal Jacket" (1987) by Hasford, Michael Herr, and Stanley Kubrick.

 Hasford's novel "The Phantom Blooper" (1990) is a sequel to "The Short-Timers".

 The book was supposed to be the first of a trilogy, but Hasford died soon after completing its sequel and before writing the third installment.
</item 8>

<item 9>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 9>

<item 10>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 10>
=== END ITEM id=e2e8fcac445b2204 ===

=== ITEM id=72f8ef6d745f129d ===
Gold sentences and facts:
Sentence 0: Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.
  Fact 0.0: Bicycling is a cycling brand.
  Fact 0.1: Bicycling is published by Rodale, Inc. in Emmaus, Pennsylvania.
Sentence 1: National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural affairs.
  Fact 1.0: National Review (NR) is an American magazine.
  Fact 1.1: National Review is semi-monthly.
  Fact 1.2: National Review is conservative.
  Fact 1.3: National Review is an editorial magazine.
  Fact 1.4: National Review focuses on news and commentary pieces on political affairs.
  Fact 1.5: National Review focuses on news and commentary pieces on social affairs.
  Fact 1.6: National Review focuses on news and commentary pieces on cultural affairs.

Retrieved items:
<item 1>
National Review Board
The National Review Board (full name: National Review Board for the Protection of Children and Young People) is a committee created in 2002 by the United States
Conference of Catholic Bishops in order to monitor the implementation of the Charter for the Protection of Children and Young People in the wake of the clerical
abuse scandal in the United States.

 The Board was also charged with investigating the scandal, which it did in part by commissioning the John Jay College to conduct a survey of Church records in
order to define the nature and scope of the abuse committed by priests over the period between 1950 and 2002.

 The results of that survey were released in 2004 in what has come to be known as the "John Jay Report" or "Nature and Scope report."

 In parallel with the John Jay College survey the Board conducted interviews with a variety of people, both inside and outside the Church, who were well placed
to comment on the scandal, and on the basis of these interviews prepared a more broad-ranging report of its own.
</item 1>

<item 2>
Charles C. W. Cooke
Charles C. W. Cooke (born November 4, 1984) is the editor of "National Review Online".

 He took the role over after Rich Lowry stepped down in June, 2016 (Lowry remains the editor-in-chief of "National Review").

 Cooke is the author of "The Conservatarian Manifesto" and a frequent guest on HBO's "Real Time with Bill Maher".

 In addition to "National Review", he has written for the "New York Times", the "Washington Post", and the "Los Angeles Times".

 Along with Kevin D. Williamson, he hosts the popular "Mad Dogs and Englishmen" podcast.
</item 2>

<item 3>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 3>

<item 4>
Review
A review is an evaluation of a publication, service, or company such as a movie (a movie review), video game (video game review), musical composition (music
review of a composition or recording), book (book review); a piece of hardware like a car, home appliance, or computer; or an event or performance, such as a
live music concert, play, musical theater show, dance show, or art exhibition.

 In addition to a critical evaluation, the review's author may assign the work a rating to indicate its relative merit.

 More loosely, an author may review current events, trends, or items in the news.

 A compilation of reviews may itself be called a review. "

The New York Review of Books", for instance, is a collection of essays on literature, culture, and current affairs. "

National Review", founded by William F. Buckley, Jr., is an influential conservative magazine, and "Monthly Review" is a long-running socialist periodical.
</item 4>

<item 5>
National Review
National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural
affairs.

 The magazine was founded by the author William F. Buckley Jr. in 1955.

 It is currently edited by Rich Lowry.
</item 5>

<item 6>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 6>

<item 7>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 7>

<item 8>
National Review Board
Whereas the John Jay College report was (as intended) primarily a factual summary of the data collected in the College’s survey of Church records, the Board’s
own report sought to interpret these data and its other findings in order to explain why the “epidemic” of clerical abuse had occurred and to identify the
appropriate steps to avoid any repetition.
</item 8>

<item 9>
Kevin D. Williamson
Kevin Daniel Williamson (born September 18, 1972) is the roving correspondent for "National Review."

 He is also the theater critic for "The New Criterion".

 He was previously deputy managing editor at "National Review".
</item 9>

<item 10>
Kate O'Beirne
Kate Walsh O'Beirne (September 23, 1949 – April 23, 2017) was the President of National Review Institute.

 She was the Washington editor of "National Review".

 Her column, "Bread and Circuses," covered Congress, politics, and U.S. domestic policy.
</item 10>
=== END ITEM id=72f8ef6d745f129d ===

=== ITEM id=214331c3338b806d ===
Gold sentences and facts:
Sentence 0: Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.
  Fact 0.0: Buried Country is a highly regarded documentary film
  Fact 0.1: Buried Country is a highly regarded book
  Fact 0.2: Buried Country is a highly regarded soundtrack album
  Fact 0.3: Buried Country is now also a stageshow
Sentence 1: South of the Border is a 2009 American documentary film directed by Oliver Stone.
  Fact 1.0: South of the Border is a documentary film
  Fact 1.1: South of the Border is American
  Fact 1.2: South of the Border is from 2009
  Fact 1.3: South of the Border was directed by Oliver Stone

Retrieved items:
<item 1>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 1>

<item 2>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 2>

<item 3>
Cinema of Uruguay
The cinema of Uruguay has a role in the culture of Uruguay and is a part of Latin American cinema.

 Since the late 1990s, the Uruguayan cinema undergoes a process of evolution, during which its films have received positive reviews and been internationally
recognized.

 Over 120 films, fictions and non-fictions, have been produced since then.
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
The Colleen Bawn
The Colleen Bawn, or The Brides of Garryowen is a melodramatic play written by Irish playwright Dion Boucicault.

 It was first performed at Miss Laura Keene's Theatre, New York, on 27 March 1860 with Laura Keene playing Anne Chute and Boucicault playing Myles na Coppaleen.

 It was most recently performed in Dublin at the Project Arts Centre in July and August 2010.

 Several film versions have also been made.
</item 5>

<item 6>
Plain Clothes Theatre Productions
Plain Clothes Theatre Productions is a Bristol-based theatre company producing contemporary plays from around the globe.

 Formed in 2003 by artistic director Sam Berger, the company has produced work in London, Bristol, Cheltenham, Toronto and Vancouver.

 Their work has included plays by Joe Orton, Laura Wade and David Mamet, and their style is based around the teachings of American acting coach Sanford Meisner.
</item 6>

<item 7>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 7>

<item 8>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 8>

<item 9>
Buried Country
Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.

 It is a prosopography, created by noted writer Clinton Walker, that tells the story of Australian country music in the Aboriginal community by focussing on the
genre's most important stars.
</item 9>

<item 10>
Station to Station (film)
Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.

 It was distributed internationally in 2015 following its release at the Sundance Film Festival.
</item 10>
=== END ITEM id=214331c3338b806d ===
