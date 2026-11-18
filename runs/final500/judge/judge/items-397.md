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

=== ITEM id=cd2e6ae54e0783ea ===
Gold sentences and facts:
Sentence 0:  It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!
  Fact 0.0: The entity referred to as 'It' is known for producing the long-running series 'Tom Goes to the Mayor'.
  Fact 0.1: The entity referred to as 'It' is known for producing the long-running series 'Nathan For You'.
  Fact 0.2: The entity referred to as 'It' is known for producing the long-running series 'Tim and Eric Awesome Show, Great Job!'.
Sentence 1: " and "Check It Out!
  Fact 1.0: 'Check It Out!' is a series title.
Sentence 2:  with Dr. Steve Brule".
  Fact 2.0: The series title is 'Check It Out! with Dr. Steve Brule'.
Sentence 3: Check It Out!
  Fact 3.0: Check It Out! is a title.
Sentence 4:  with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!
  Fact 4.0: Check It Out! with Dr. Steve Brule is an American sketch comedy television series.
  Fact 4.1: Check It Out! with Dr. Steve Brule is a spin-off of 'Tim and Eric Awesome Show, Great Job!'.

Retrieved items:
<item 1>
Abso Lutely Productions
Abso Lutely Productions is a film and television production company owned by actor and producers Tim Heidecker, Eric Wareheim and producer Dave Kneebone.

 It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!

" and "Check It Out!

 with Dr. Steve Brule".
</item 1>

<item 2>
The collection profiles entertainers, creators, and fictional characters across film, television, music, comedy, and magic. Highlights include magician Criss
Angel’s Las Vegas shows; actors such as Michael Crawford, Matthew Perry, Josh Charles, and Glenn Quinn; comedians and creators behind *In Living Color*,
*Upright Citizens Brigade*, and *Family Guy*; and performers including Nat and Alex Wolff and MatPat. It also covers productions such as *American Pie Presents:
Band Camp* and *
</item 2>

<item 3>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
</item 3>

<item 4>
Check It Out! with Dr. Steve Brule
Check It Out!

 with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!

" starring John C. Reilly as Dr. Steve Brule.

 The series premiered on Cartoon Network's late night programming block, Adult Swim, on May 16, 2010.

 The program follows Brule as he examines different facets of living.

 His severe naivete and social awkwardness generally land him in embarrassing situations, though he largely remains ignorant of any embarrassment he's causing
himself.

 As the series progresses, he reveals shocking and sometimes horrifying details about his past and personal life.
</item 4>

<item 5>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 5>

<item 6>
Upright Citizens Brigade (TV series)
Upright Citizens Brigade is an American sketch comedy television series that premiered on August 19, 1998 on Comedy Central.

 The show aired for three seasons with each season consisting of ten episodes.

 The series featured four members of Upright Citizens Brigade, an improvisational sketch comedy group.

 The cast included Matt Besser, Amy Poehler, Ian Roberts, and Matt Walsh.

 The cast would later reunite for another series of a similar format that premiered in 2016 on Seeso.
</item 6>

<item 7>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 7>

<item 8>
Bagboy (TV special)
Bagboy is a 2015 American television special produced for Adult Swim, and aired on February 21, 2015, to positive critical reception.

 Written and directed by Tim Heidecker and Eric Wareheim for Abso Lutely Productions and John C. Reilly, the special is a fictional sitcom pilot in the universe
of "Tim and Eric Awesome Show, Great Job!

", written by and starring Reilly's recurring "Tim and Eric" character Dr. Steve Brule.

 This is the second "Tim and Eric" spin-off to feature the Dr. Steve Brule character after "Check it Out!

 with Dr. Steve Brule".
</item 8>

<item 9>
HGTV Design Star S3 (2008): Katrina-home finale; Jennifer Bertrand won a viewer vote and her own show. Friends of the People (2014): TruTV’s first sketch
series, with 10 episodes and former In Living Color revival cast members. In Living Color (1990–94): Wayans-created Fox series, notable for Black-centered humor
and disputes with executives. Key & Peele (2012–15): 53 episodes across five seasons, plus a Super
</item 9>

<item 10>
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 10>
=== END ITEM id=cd2e6ae54e0783ea ===

=== ITEM id=fec440afe107b307 ===
Gold sentences and facts:
Sentence 0: Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.
  Fact 0.0: Frederick King Keller was born in 1954.
  Fact 0.1: Frederick King Keller was born in Buffalo, New York.
  Fact 0.2: Frederick King Keller is an American director.
  Fact 0.3: Frederick King Keller is an American producer.
  Fact 0.4: Frederick King Keller is an American screenwriter.
  Fact 0.5: Frederick King Keller works in film and television.
Sentence 1: Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
  Fact 1.0: Ian Schneider was born on October 3, 1992.
  Fact 1.1: Ian Schneider is an independent American film director.
  Fact 1.2: Ian Schneider is an American screenwriter.

Retrieved items:
<item 1>
Colby Keller
Colby Keller (born October 18, 1980) is an American visual artist, blogger, and pornographic film actor.

 His career in porn film started in 2004 at Sean Cody and has since expanded to include such studios as Cocksure Men, Randy Blue, Titan Men, Falcon, CockyBoys
and Men.com.

 Keller has amassed a large fan base with both his films and his long-standing blog, "The Big Shoe Diaries".

 He has been nominated for a number of awards and in 2013, World of Wonder awarded him the WOWie Award for Best Blog.

 Keller is nominated for 8 2015 Grabby Awards.

 Keller has also appeared in the short, Zolushka, a queer retelling of Cinderella and the popular series Capitol Hill both by the cult filmmaker Wes Hurley.
</item 1>

<item 2>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 2>

<item 3>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 3>

<item 4>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 4>

<item 5>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 5>

<item 6>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 6>

<item 7>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 7>

<item 8>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 8>

<item 9>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 9>

<item 10>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 10>
=== END ITEM id=fec440afe107b307 ===

=== ITEM id=def0f1a0204c6597 ===
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
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
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
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 4>

<item 5>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 5>

<item 6>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 6>

<item 7>
Shanxi–Henan–Shandong Railway
The line can accommodate train speeds of up to 120 km/h .

 Cities along route include Lüliang, Liulin, Hongdong, Changzhi and Pingshun in Shanxi; Linzhou, Anyang, Tangyin, Puyang, Hebi, Fan County and Taiqian County in
Henan; Liangshan County, Dongping, Ningyang, Xintai, Laiwu, Yiyuan, Yishui, Ju County, Junan and Rizhao in Shandong.

 The line was built from 2010 to 2014 and operations began on December 30, 2014.
</item 7>

<item 8>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 8>

<item 9>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 9>

<item 10>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 10>
=== END ITEM id=def0f1a0204c6597 ===

=== ITEM id=75cc7b786e1c7984 ===
Gold sentences and facts:
Sentence 0: Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on the German island of Greifswalder Oie in December 1937.
  Fact 0.0: Operation Lighthouse was the name given to a failed experimental launch.
  Fact 0.1: The Operation Lighthouse launch involved four Aggregate 3 liquid-fuel rockets.
  Fact 0.2: Wernher von Braun conducted the Operation Lighthouse launch.
  Fact 0.3: Walter Dornberger conducted the Operation Lighthouse launch.
  Fact 0.4: The Operation Lighthouse launch took place on the German island of Greifswalder Oie.
  Fact 0.5: The Operation Lighthouse launch took place in December 1937.
Sentence 1: Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War II.
  Fact 1.0: Major-General Dr. Walter Robert Dornberger was born on 6 September 1895.
  Fact 1.1: Walter Robert Dornberger died on 27 June 1980.
  Fact 1.2: Walter Robert Dornberger was a German Army artillery officer.
  Fact 1.3: Walter Robert Dornberger's career spanned World War I.
  Fact 1.4: Walter Robert Dornberger's career spanned World War II.

Retrieved items:
<item 1>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 1>

<item 2>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 2>

<item 3>
Von Braun (crater)
von Braun, named after the rocket pioneer Wernher von Braun, is a lunar crater located near the northwestern limb of the Moon.

 It lies along the western edge of the Oceanus Procellarum, to the northeast of the crater Lavoisier.

 The northeastern rim of this crater is on the edge of the Sinus Roris, a bay feature in the northwestern part of the Oceanus Procellarum.

 Due to its proximity to the limb, this crater appears significantly foreshortened when viewed from the Earth.
</item 3>

<item 4>
Man Will Conquer Space Soon!
Man Will Conquer Space Soon!

 was the title of a famous series of 1950s magazine articles in "Collier's" detailing Wernher von Braun's plans for manned spaceflight.

 Edited by Cornelius Ryan, the individual articles were authored by such space notables of the time as Willy Ley, Fred Lawrence Whipple, Dr. Joseph Kaplan, Dr.
Heinz Haber, and von Braun.

 The articles were illustrated with paintings and drawings by Chesley Bonestell, Fred Freeman, and Rolf Klep, some of the finest magazine illustrators of the
time.
</item 4>

<item 5>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 5>

<item 6>
Von Braun Ferry Rocket
Von Braun Ferry Rocket was a concept design for a shuttle spacecraft that was developed by Wernher von Braun in a seminal series of early-1950s Collier's
magazine articles, Man Will Conquer Space Soon!

 by Wernher von Braun "et al." The Ferry Rocket concept has evolved over time.
</item 6>

<item 7>
Magnus von Braun
Magnus "Mac" Freiherr von Braun (10 May 1919 – 21 June 2003) was a German chemical engineer, Luftwaffe aviator, and rocket scientist at Peenemünde, the
Mittelwerk, and after emigrating to the United States via Operation Paperclip, at Fort Bliss.

 He was the brother of Sigismund and Wernher von Braun.
</item 7>

<item 8>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 8>

<item 9>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 9>

<item 10>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 10>
=== END ITEM id=75cc7b786e1c7984 ===

=== ITEM id=3468bb862879a8dd ===
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
Volvo P2 platform
The Volvo P2 platform is a global full-size unibody automobile platform developed and produced by Volvo.

 It is designed for single wheelbases and is adaptable to front- or all wheel drive configurations.

 It was developed by the automaker before its 1999 acquisition by Ford Motor Company, and debuted with the 1998 Volvo S80.
</item 1>

<item 2>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 2>

<item 3>
Volvo 164
The Volvo 164 is a 4-door, 6-cylinder luxury sedan unveiled by Volvo at the Paris Motor Show early in October 1968 and first sold as a 1969 model.

 46,008 164s were built before the car was superseded by the 264 in 1975.

 The 164 was Volvo's first venture into the luxury segment since the end of PV 60 production in 1950, and was the first six-cylinder Volvo since the PV800 last
produced in 1958.
</item 3>

<item 4>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 4>

<item 5>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 5>

<item 6>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 6>

<item 7>
Volvo V50
"For the sedan variant of the car, see Volvo S40."
</item 7>

<item 8>
Volvo SI6 engine
Volvo designed the SI6 ("short inline 6") straight-6 automobile engine for use in 2007 models.

 An evolution of the company's long-used straight-5 Volvo Modular engine, which itself is an evolution of the Volvo B6304 straight six engine, the SI6 can be
mounted transversely for front wheel drive or all wheel drive applications.

 Despite the added cylinder and displacement, the engine remains compact, and is in fact 1 mm shorter than the previous straight-5.

 The engine was initially offered in two displacements — a 3.0 L turbocharged version and a 3.2 L naturally aspirated version.

 Both offered variable cam timing, though only the turbo version varies both the intake and exhaust valves.

 On top of the variable cam timing used on the intake cam of the naturally aspirated engine it also had variable valve lift by using Cam Profile Switching
(CPS).
</item 8>

<item 9>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 9>

<item 10>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 10>
=== END ITEM id=3468bb862879a8dd ===

=== ITEM id=15ba088807e9a431 ===
Gold sentences and facts:
Sentence 0:  It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
  Fact 0.0: The work is the third installment of the 'Naked' series.
  Fact 0.1: 'Naked Killer' is a work in the 'Naked' series released in 1992.
  Fact 0.2: 'Naked Weapon' is a work in the 'Naked' series released in 2002.
Sentence 1: Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
  Fact 1.0: Naked Weapon is also titled 赤裸特工.
  Fact 1.1: Naked Weapon is a 2002 film.
  Fact 1.2: Naked Weapon is a Hong Kong film.
  Fact 1.3: Naked Weapon is an action-thriller film.
  Fact 1.4: Naked Weapon was directed by action choreographer Tony Ching.
  Fact 1.5: Naked Weapon stars Maggie Q.
  Fact 1.6: Naked Weapon stars Anya Wu.
  Fact 1.7: Naked Weapon stars Daniel Wu.

Retrieved items:
<item 1>
The Naked Gun 2½: The Smell of Fear
The Naked Gun 2½: The Smell of Fear is a 1991 comedy film.

 It is the sequel to the 1988 film "" and the second installment in "The Naked Gun" film series.

 The film stars Leslie Nielsen as the comically bumbling Police Lt. Frank Drebin of "Police Squad!

".

 Priscilla Presley plays the role of Jane, with O. J. Simpson as Nordberg and George Kennedy as police captain Ed Hocken.

 The film also features Robert Goulet (who previously made a "special guest star" appearance on "Police Squad!")

 as the villainous Quentin Hapsburg and Richard Griffiths as renewable fuel advocate Dr. Albert S. Meinheimer (as well as his evil double, Earl Hacker).

 Zsa Zsa Gabor, Mel Tormé and members of the Chicago Bears have cameo roles.
</item 1>

<item 2>
Naked Gun 33⅓: The Final Insult
Naked Gun 33⅓: The Final Insult is a 1994 comedy film, and the third and final installment in "The Naked Gun" film series, which was based on the "Police Squad!

" television series.
</item 2>

<item 3>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 3>

<item 4>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 4>

<item 5>
Naked Soldier
Naked Soldier (絕色武器) is a 2012 Hong Kong action film directed by Marco Mak and starring Jennifer Tse.

 It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
</item 5>

<item 6>
Mel Gibson filmography
Five years later he played Martin Riggs in the buddy cop action comedy "Lethal Weapon" alongside Danny Glovera role he later reprised in its sequels "Lethal
Weapon 2" (1989), "Lethal Weapon 3" (1992), and "Lethal Weapon 4" (1998).
</item 6>

<item 7>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 7>

<item 8>
Naked Weapon
Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
</item 8>

<item 9>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 9>

<item 10>
American Pie Presents: The Naked Mile
American Pie Presents: The Naked Mile (also known as American Pie: The Naked Mile) is a 2006 American sex comedy film released by Universal Pictures.

 It is the second installment in the "American Pie Presents" series and the fifth installment in the "American Pie" franchise.

 The film begins a story arc that concludes with "" (2007).

 John White stars as Erik Stifler, a high school senior who is given a "guilt free pass" by his girlfriend, Tracy Sterling (Jessy Schram), and so visits the
Beta House fraternity led by his cousin, Dwight Stifler (Steve Talley), to run a mile naked.

 Christopher McDonald co-stars as Erik's father, Harry, and Eugene Levy once again plays Jim's Dad, who turns out to be a family friend of both Erik's and
Tracy's.

 Also, it is in this film that his name is revealed to be "Noah Levenstein".
</item 10>
=== END ITEM id=15ba088807e9a431 ===

=== ITEM id=8715f91dfa89e942 ===
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
Rachelle Henry
Rachelle Henry (born December 16, 2000) is an American teen actress and filmmaker.

 She played the role of Sandy Hobbs in the TLC (TV network) Series "Escaping the Prophet"and Lissa Golaski in "Depth", the film prequel to the Soma (video game)
by Frictional Games.

 She is also known for directing and producing short films containing messages of social influence and coming of age themes including "MISSING" and "Almost
Boyfriends".
</item 1>

<item 2>
Rachelle Henry (born 2000) is an American teen actress and filmmaker, known for *Escaping the Prophet*, *Depth*, and directing socially themed shorts *MISSING*
and *Almost Boyfriends*. British actress and stunt performer Rachelle Beinart (born 1984) worked on *Game of Thrones*’ award-winning stunt teams, *Rogue One*,
and *Teletubbies*. Paula Patton (born 1975) is an American actress whose films include
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
The passage profiles actresses, voice performers, filmmakers, singers, and pageant figures. Highlights include Rachelle Henry’s acting and socially themed short
films; Rachelle Lefevre’s *Twilight* role; Eva Gabor’s *Green Acres* and Disney voice work; Andrea Robinson’s singing in *Sister Act*; and Kelly Metzger’s
animation roles. It also notes Amy O’Neill in *Honey, I Shrunk the Kids*, Heather Burns in *Miss Cong
</item 4>

<item 5>
Amy O’Neill (born 1971) played Amy Szalinski in *Honey, I Shrunk the Kids* (1989) and its 1992 sequel, earning a Young Artist Award nomination. *Rhoda*, Valerie
Harper’s *Mary Tyler Moore Show* spin-off, ran five seasons (1974–78), followed Rhoda back to New York, and won two Golden Globes and two Emmys. Allison Miller
(born 1985) starred in *K
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
Rachelle Beinart
Rachelle Amy Beinart (born 29 September 1984) is a British actress and stunt performer.

 She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".

 She has also acted in preschool children's shows as a suit actor, including the reboot of "Teletubbies".
</item 7>

<item 8>
Rogue One
Rogue One: A Star Wars Story, or simply Rogue One, is a 2016 American epic space opera film directed by Gareth Edwards.

 The screenplay by Chris Weitz and Tony Gilroy is from a story by John Knoll and Gary Whitta.

 It was produced by Lucasfilm and distributed by Walt Disney Studios Motion Pictures.

 It is the first installment of the "Star Wars" "Anthology" series, set immediately before the events of the original "Star Wars" film.

 The cast includes Felicity Jones, Diego Luna, Riz Ahmed, Ben Mendelsohn, Donnie Yen, Mads Mikkelsen, Alan Tudyk, Jiang Wen and Forest Whitaker.

 "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
</item 8>

<item 9>
The entries profile filmmakers, producers, performers, and artists: Tim Bedore is a Chicago-born comedian; Rogue One (2016) depicts rebels stealing Death Star
plans; Frederick K. Keller, Peter Guber, Baltasar Kormákur, Steve Rash, Jonathan Kaplan, Kurt Voss, Jim Sharman, Barry Sandler, and Howard Rosenman are
directors or producers. Other highlights include Titan A.E.’s commercial failure; Vito Russo’s documentary; Scott Free’s founding by the
</item 9>

<item 10>
Canadian actress Rachelle Lefevre starred in *Big Wolf on Campus*, played Victoria in the first two *Twilight* films, and appeared in *Off the Map*, *A Gifted
Man*, and *Under the Dome*. American actress Gloria Garayua debuted in *Fun with Dick and Jane* and had roles in *Grey’s Anatomy*, *Cougar Town*, and *How to
Get Away with Murder*. Zooey Deschanel’s films include *Almost Famous*, *All
</item 10>
=== END ITEM id=8715f91dfa89e942 ===

=== ITEM id=1999a2f26ed9f6ce ===
Gold sentences and facts:
Sentence 0: 888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.
  Fact 0.0: 888 7th Avenue is a 628 ft (191 m) tall modern-style office skyscraper.
  Fact 0.1: 888 7th Avenue is in Midtown Manhattan.
  Fact 0.2: 888 7th Avenue was completed in 1969.
  Fact 0.3: 888 7th Avenue has 46 floors.
Sentence 1: 200 West Street is the global headquarters of the Goldman Sachs investment banking firm.
  Fact 1.0: 200 West Street is the global headquarters of the Goldman Sachs investment banking firm.
Sentence 2:  The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.
  Fact 2.0: The building is 749 ft tall.
  Fact 2.1: The building has 44 stories.
  Fact 2.2: The building is located on West Street.
  Fact 2.3: The building is located between Vesey Street and Murray Street.
  Fact 2.4: The building is in Lower Manhattan.

Retrieved items:
<item 1>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 1>

<item 2>
The entries cover notable U.S. buildings, districts, businesses, and people. New York features prominently: office towers such as 888 Seventh Avenue, 245 Park
Avenue, 383 and 270 Madison/Park Avenue, and Goldman Sachs’ 200 West Street; landmarks including the Woolworth Building and 90 West Street; and residential or
mixed-use towers such as MiMA and Random House Tower. Other topics include historic theaters, retail and media companies, transportation firms, and real-estate
developer
</item 2>

<item 3>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 3>

<item 4>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
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
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 6>

<item 7>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
</item 7>

<item 8>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 8>

<item 9>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 9>

<item 10>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 10>
=== END ITEM id=1999a2f26ed9f6ce ===

=== ITEM id=bf95c3c5a0144f64 ===
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
</item 1>

<item 2>
Marquette Transportation Company
The Marquette Transportation Company is a marine transportation company based in Paducah, Kentucky, United States.
</item 2>

<item 3>
Toland, Pennsylvania
Toland is less than 1/4 of a mile long, and it is located less than half a mile from the Appalachian trail crossing.

 The closest town is Mount Holly Springs, Pennsylvania located (3 miles north of Toland).

 The village is roughly 10 miles south of Carlisle, Pennsylvania, 7 miles south east of Boiling Springs, Pennsylvania, and 15 miles South of Gettysburg,
Pennsylvania.

 The only business in Toland is the Cherokee Campground formerly known as the Tagg Run Campground.

 The campground restaurant is now closed.

 Toland Mission is a small non-denominational church that can hold up to 74 persons.

 It was originally built by the owner of Beetem Lumber Company in Carlisle for the families of the community of Toland.

 A one-room school house near the church originally served the community, but it was converted to a home when residents' children were transported to a township
school.

 Local people in Toland usually travel to the town of Mount Holly Springs for necessities such as gas, food, and toiletries.
</item 3>

<item 4>
Transportation in Jacksonville, Florida
The Jacksonville transportation network includes ground, air, and sea options for passenger and freight transit.

 The Jacksonville Port Authority (Jaxport) operates the Port of Jacksonville, which includes container shipping facilities at "Blount Island Marine Terminal",
the "Talleyrand Marine Terminal" and the "Dames Point Marine Terminal".

 Jacksonville Aviation Authority managers Jacksonville International Airport in Northside, as well as several smaller airports.

 The Jacksonville Transportation Authority (JTA) operates bus, people mover, and park-n-ride services throughout the city and region.

 A major bus terminal at the intermodal Rosa Parks Transit Station serves as JTA's main transit hub.

 Various intercity bus companies terminate near Central Station.

 Amtrak operates passenger rail service to and from major cities throughout North America.

 The city is bisected by major highways, I-95 and I-10, I-295 creates a full beltway around the city.
</item 4>

<item 5>
Merchants Transportation Company
The Merchants Transportation Company was a shipping firm that operated on Puget Sound from 1905 to 1929.

 This company should not be confused with the similarly named Merchants Transportation Company of Olympia, formed in 1874 and a completely separate firm.
</item 5>

<item 6>
Northern Transportation Company
Northern Transportation Company Limited (NTCL) is a marine transportation company in the Canadian and American Arctic owned by Norterra, a holding company that
is owned by the Inuvialuit of the Northwest Territories.

 While primarily a marine freight hauler on Canadian rivers, like the Mackenzie and the Hay and along the Arctic coast of Canada, they are also a petroleum
wholesaler.

 Its head office is now in the town of Hay River, Northwest Territories.

 Today, it primarily utilizes tugs and barges.
</item 6>

<item 7>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 7>

<item 8>
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 8>

<item 9>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 9>

<item 10>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 10>
=== END ITEM id=bf95c3c5a0144f64 ===

=== ITEM id=60cce9249c4ab6d7 ===
Gold sentences and facts:
Sentence 0: Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the Himalayas, and Indonesia) and the Americas.
  Fact 0.0: Hydrangea is a genus of flowering plants.
  Fact 0.1: Hydrangea's common names are hydrangea or hortensia.
  Fact 0.2: The Hydrangea genus has 70 to 75 species.
  Fact 0.3: Hydrangea plants are native to southern and eastern Asia.
  Fact 0.4: Hydrangea plants are native to China, Japan, Korea, the Himalayas, and Indonesia.
  Fact 0.5: Hydrangea plants are native to the Americas.
Sentence 1:  It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").
  Fact 1.0: The subject of this sentence contains 13 to 18 species.
  Fact 1.1: The subject of this sentence is best known for a single species.
  Fact 1.2: That single species is Jequirity.
  Fact 1.3: Jequirity is also called A. precatorius.

Retrieved items:
<item 1>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 1>

<item 2>
Hydrangea serrata
Hydrangea serrata is a species of flowering plant in the family Hydrangeaceae, native to mountainous regions of Korea and Japan.

 Common names include mountain hydrangea and tea of heaven.

 Growing to 1.2 m tall and broad, it is a deciduous shrub with oval leaves and panicles of blue and pink flowers in summer and autumn (fall).

 It is widely cultivated as an attractive ornamental shrub throughout the world in areas with suitable climate and soil.
</item 2>

<item 3>
Hydrangea arborescens
Hydrangea arborescens, commonly known as smooth hydrangea, wild hydrangea, or sevenbark, is a species of flowering plant in the family Hydrangeaceae.

 It is a small- to medium-sized, deciduous shrub up to 3 m tall that is native to the eastern United States.
</item 3>

<item 4>
Hydrangea paniculata
Hydrangea paniculata, the panicled hydrangea, is a species of flowering plant in the family Hydrangeaceae native to southern and eastern China, Korea, Japan and
Russia (Sakhalin).

 It was first formally described by Philipp Franz von Siebold in 1829.

 It is a deciduous shrub or small tree, 1 - tall by 2.5 m broad, growing in sparse forests or thickets in valleys or on mountain slopes.
</item 4>

<item 5>
Hydrangea macrophylla
Hydrangea macrophylla is a species of flowering plant in the family Hydrangeaceae, native to Japan.

 It is a deciduous shrub growing to 2 m tall by 2.5 m broad with large heads of pink or blue flowers in summer and autumn.

 Common names include bigleaf hydrangea, French hydrangea, lacecap hydrangea, mophead hydrangea, penny mac and hortensia.

 It is widely cultivated in many parts of the world in many climates.

 It is not to be confused with "H. aspera" 'Macrophylla'.
</item 5>

<item 6>
Hydrangea quercifolia
Hydrangea quercifolia, commonly known by its translation oakleaf hydrangea or oak-leaved hydrangea, is a species of flowering plant native to the Southeastern
United States, in woodland habitats from North Carolina west to Tennessee, and south to Florida and Louisiana.

 A deciduous shrub with white showy flower heads, it is a commonly grown garden plant.

 Numerous cultivars are available commercially.
</item 6>

<item 7>
Abrus canescens
Abrus canescens is a species of flowering plant belonging to the legume family, native to Africa.

 It is considered almost extinct.
</item 7>

<item 8>
Hydrangea heteromalla
Hydrangea heteromalla is a species of "Hydrangea" flower, native to the Himalayas and China.

 Average height is 10'-15' with inflorescences that are 5"-8" across and bloom May-June,full flowering only occurs in full sun.Is a very adaptable plant that is
hardy to USDA zones (4)5-7.

 This plant is seldom seen in American gardens although there is nice specimen at the Hillier Arboretum.
</item 8>

<item 9>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 9>

<item 10>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 10>
=== END ITEM id=60cce9249c4ab6d7 ===

=== ITEM id=de8bcdb308c71e5e ===
Gold sentences and facts:
Sentence 0: Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.
  Fact 0.0: Age of Chance was a British alternative rock-dance crossover band.
  Fact 0.1: Age of Chance was from Leeds, England.
  Fact 0.2: Age of Chance was active from 1985 to 1991.
Sentence 1: Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.
  Fact 1.0: Vector was a rock band.
  Fact 1.1: Vector was formed in Sacramento, California.
  Fact 1.2: Vector was formed in the early 1980s.
  Fact 1.3: Vector was formed by Jimmy Abegg.
  Fact 1.4: Vector was formed by Steve Griffith.
  Fact 1.5: Vector was formed by Charlie Peacock.

Retrieved items:
<item 1>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 1>

<item 2>
William Gibson
William Ford Gibson (born March 17, 1948) is an American-Canadian speculative fiction writer and essayist widely credited with pioneering the science fiction
subgenre known as cyberpunk.

 Beginning his writing career in the late 1970s, his early works were noir, near-future stories that explored the effects of technology, cybernetics, and
computer networks on humans—a "combination of lowlife and high tech"—and helped to create an iconography for the information age before the ubiquity of the
Internet in the 1990s.

 Gibson notably coined the term "cyberspace" in his short story "Burning Chrome" (1982) and later popularized the concept in his acclaimed debut novel
"Neuromancer" (1984).

 These early works have been credited with "renovating" science fiction literature after it had fallen largely into insignificance in the 1970s.
</item 2>

<item 3>
Word2vec
Word2vec is a group of related models that are used to produce word embeddings.

 These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.

 Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus
being assigned a corresponding vector in the space.

 Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the
space.
</item 3>

<item 4>
Vector (band)
Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.

 The band had several drummers over the years, including Aaron Smith and Bruce Spencer, both of whom also played drums for The 77s.
</item 4>

<item 5>
Age of Chance
Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.

 They were perhaps most known for their mutant metallic cover of Prince's "Kiss" which topped the UK Indie Chart in 1986, and peaked at No. 50 in the UK Singles
Chart in January the following year.

 Despite signing for major label Virgin, and being favourites with the UK music press, they never enjoyed a major hit in the UK, although "Don't Get Mad… Get
Even" reached No. 8 on the US "Billboard" Hot Dance/Club Play chart.
</item 5>

<item 6>
Multivariate random variable
In probability, and statistics, a multivariate random variable or random vector is a list of mathematical variables each of whose value is unknown, either
because the value has not yet occurred or because there is imperfect knowledge of its value.

 The individual variables in a random vector are grouped together because they are all part of a single mathematical system — often they represent different
properties of an individual statistical unit.

 For example, while a given person has a specific age, height and weight, the representation of these features of "an unspecified person" from within a group
would be a random vector.

 Normally each element of a random vector is a real number.
</item 6>

<item 7>
Viceland (Canada)
Viceland is a Canadian English language Category A specialty channel.

 Owned by Rogers Media, it is a Canadian version of Vice Media's Viceland network, and broadcasts lifestyle-oriented documentary and reality series aimed
towards a young adult demographic.
</item 7>

<item 8>
Christopher Evans (author)
Christopher Evans (born 1951 in Tredegar, Wales) is a British science fiction writer and children's author.

 His novels include "Capella's Golden Eyes" (1980), "The Insider" (1981), "Mortal Remains" (1995) and "Ice Tower" (2000).

 He is the co-editor (with Robert Holdstock) of three original SF anthologies, "Other Edens" (1987), "Other Edens II" (1988) and "Other Edens III" (1989).
</item 8>

<item 9>
The Four Winds of Love
The Four Winds of Love is the overall title for a series of six novels written by Compton Mackenzie, "The East Wind of Love" (1937), "The South Wind of Love"
(1938), "The West Wind of Love" (1940), "West to North" (1942), "The North Wind of Love, Book 1" (1944) and "The North Wind of Love, Book 2" (1945), which taken
together constitute a major fictional chronicle of the first forty years of the twentieth century.

 The main protagonist of the hexalogy is the semi-autobiographical character of John Ogilvie.
</item 9>

<item 10>
New Valley LLC
New Valley LLC is an investment company owned by Vector Group.

 Western Union was renamed New Valley in the early 1990s as the corporate parent of all Western Union related businesses.

 In 1994, New Valley sold its Western Union businesses to First Financial Management (later acquired by First Data) for $1.2 Billion.

 After more than 100 years of being a publicly traded entity, New Valley became a wholly owned subsidiary of Vector Group on December 13, 2005 when Vector Group
acquired the remaining 42.3% of New Valley's common shares that it did not already own.

 New Valley presently has a 70% stake in Douglas Elliman Realty, the largest residential real estate brokerage in the New York metropolitan area.

 The company also has stakes in various hotels and golf courses in the United States.

 In conjunction with the spin-off of its Ladenburg Thalmann subsidiary New Valley retained a 7.6% stake in Ladenburg Thalmann Financial Services, an investment
bank and stock brokerage based in Miami.
</item 10>
=== END ITEM id=de8bcdb308c71e5e ===

=== ITEM id=4503ac2e841433d3 ===
Gold sentences and facts:
Sentence 0: Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.
  Fact 0.0: The Cat in the Hat is a Dr. Seuss film.
  Fact 0.1: The Cat in the Hat is a 2003 film.
  Fact 0.2: The Cat in the Hat is an American film.
  Fact 0.3: The Cat in the Hat is a family comedy film.
  Fact 0.4: The Cat in the Hat was directed by Bo Welch.
Sentence 1: Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
  Fact 1.0: Robert W. 'Bo' Welch III was born November 30, 1951.
  Fact 1.1: Robert W. 'Bo' Welch III is an American production designer.
  Fact 1.2: Robert W. 'Bo' Welch III is an American director.

Retrieved items:
<item 1>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 1>

<item 2>
The Cat in the Hat (film)
Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.

 It is based on the 1957 Dr. Seuss book "of the same name".

 The film stars Mike Myers in the title role of the Cat in the Hat, and Dakota Fanning as Sally.

 Sally's brother (who is unnamed in the book and the 1971 TV special), Conrad, is portrayed by Spencer Breslin.

 The film is the second feature-length Dr. Seuss adaptation after the 2000 holiday film "How the Grinch Stole Christmas".
</item 2>

<item 3>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 3>

<item 4>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 4>

<item 5>
Bo Welch
Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
</item 5>

<item 6>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 6>

<item 7>
Nitish Roy
Nitish Roy is an Indian film art director, production designer, and costume designer in Hindi cinema and a Bengali Film Director, who is also known for his work
with art cinema directors, Shyam Benegal, Mrinal Sen and Govind Nihlani, Hindi mainstream cinema, directors like Rajkumar Santoshi, as well as international
directors like Mira Nair and Gurinder Chadha.
</item 7>

<item 8>
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
</item 8>

<item 9>
Fritz the Cat (film)
Fritz the Cat is a 1972 American adult animated comedy film written and directed by Ralph Bakshi.

 It was Bakshi's feature film debut and is loosely based on the Fritz the Cat comic strips by Robert Crumb.

 It was the first animated feature film to receive an X rating in the United States.
</item 9>

<item 10>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 10>
=== END ITEM id=4503ac2e841433d3 ===
