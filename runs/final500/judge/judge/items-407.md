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

=== ITEM id=04c51f07c045477e ===
Gold sentences and facts:
Sentence 0: The Segugio Italiano ] is an Italian breed of dog of the scenthound family.
  Fact 0.0: The Segugio Italiano is an Italian breed of dog.
  Fact 0.1: The Segugio Italiano belongs to the scenthound family.
Sentence 1: The Sapsali (삽살이) is a shaggy Korean breed of dog.
  Fact 1.0: The Sapsali is a shaggy Korean breed of dog.
  Fact 1.1: The Sapsali's Korean name is 삽살이.
Sentence 2:  Traditionally, these dogs were believed to dispel ghosts and evil spirits.
  Fact 2.0: Traditionally, Sapsali dogs were believed to dispel ghosts.
  Fact 2.1: Traditionally, Sapsali dogs were believed to dispel evil spirits.

Retrieved items:
<item 1>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 1>

<item 2>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 2>

<item 3>
Segugio Italiano
The Segugio Italiano ] is an Italian breed of dog of the scenthound family.

 It comes in both short-haired and wire-haired varieties.

 It is thought to be an ancient breed, descended in pre-Roman eras from progenitor scenthounds in ancient Egypt.

 In 2009 ENCI (the Italian Kennel Club) registered 4,500 specimens of the short-haired variety and 1,740 wire-haired specimens, making this one of the top ten
breeds in Italy.
</item 3>

<item 4>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 4>

<item 5>
Sapsali
The Sapsali (삽살이) is a shaggy Korean breed of dog.

 The word is followed in Korean by either "gae" (meaning "dog") or the suffix "ee"/"i", but is most commonly romanized as "Sapsaree".

 Traditionally, these dogs were believed to dispel ghosts and evil spirits.
</item 5>

<item 6>
Santa's Little Helper
Santa's Little Helper is a recurring character in the American animated television series "The Simpsons".

 He is the pet greyhound of the Simpson family.

 The dog was introduced in the first episode of the show, the 1989 Christmas special "Simpsons Roasting on an Open Fire", in which his owner abandons him for
finishing last in a greyhound race.

 Homer Simpson and his son Bart, who are at the race track in hope of winning some money for Christmas presents, see this and decide to adopt the dog.
</item 6>

<item 7>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 7>

<item 8>
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 8>

<item 9>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 9>

<item 10>
Chiribaya Dog
The Chiribaya Dog (Spanish: "perro Chiribaya" ) or Peruvian shepherd dog ("perro pastor Peruano ") was a pre-Columbian breed of dog from the southwest of Peru,
identified by the 42 mummies discovered by anthropologist Sonia Guillén Oneglio in the Ilo District, Moquegua Region, on the south coast of Peru.

 It has been established that it was a llama herding dog.

 The dogs were not only an important part of the social structure of the ancient Peruvians, but they received special treatment after death as well.

 The dog variety has been referred to in various Spanish-language documentaries under different terms, such as "el perro pastor Chribaya" ('the Chiribaya
shepherd dog') and "pastor Peruano " ('Peruvian shepherd'), though the ancient Peruvians did not keep sheep.

 Its original name is unknown.
</item 10>
=== END ITEM id=04c51f07c045477e ===

=== ITEM id=e53bb15fa9da4345 ===
Gold sentences and facts:
Sentence 0: Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.
  Fact 0.0: Robin R. Bottin was born on April 1, 1959.
  Fact 0.1: Robin R. Bottin is an American special make-up effects creator.
Sentence 1:  Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing", with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".
  Fact 1.0: Robin R. Bottin is known for collaborations with directors John Carpenter, Paul Verhoeven and David Fincher.
  Fact 1.1: Robin R. Bottin worked with John Carpenter on “The Fog”.
  Fact 1.2: Robin R. Bottin worked with John Carpenter on “The Thing”.
  Fact 1.3: Robin R. Bottin worked with Paul Verhoeven on “RoboCop”.
  Fact 1.4: Robin R. Bottin worked with Paul Verhoeven on “Total Recall”.
  Fact 1.5: Robin R. Bottin worked with Paul Verhoeven on “Basic Instinct”.
  Fact 1.6: Robin R. Bottin worked with David Fincher on “Se7en”.
  Fact 1.7: Robin R. Bottin worked with David Fincher on “Fight Club”.
Sentence 2: David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.
  Fact 2.0: David Andrew Leo Fincher was born on August 28, 1962.
  Fact 2.1: David Andrew Leo Fincher is an American director and producer.
  Fact 2.2: David Andrew Leo Fincher is notably known for films, television series and music videos.
Sentence 3:  He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).
  Fact 3.0: David Fincher was nominated for the Academy Award for Best Director for “The Curious Case of Benjamin Button” (2008).
  Fact 3.1: David Fincher was nominated for the Academy Award for Best Director for “The Social Network” (2010).
Sentence 4:  For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
  Fact 4.0: For “The Social Network”, David Fincher won the Golden Globe Award for Best Director.
  Fact 4.1: For “The Social Network”, David Fincher won the BAFTA Award for Best Direction.

Retrieved items:
<item 1>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 1>

<item 2>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 2>

<item 3>
Radha Bharadwaj
Radha Bharadwaj is an Indian filmmaker, film producer and screenwriter.

 She moved to the United States in her late teens to study film.

 Bharadwaj's screenwriting and directing feature debut is "Closet Land".

 The surreal psychological drama was released by Universal Pictures in 1991, making Radha Bharadwaj the first director of Indian descent to have a film released
by a major Hollywood studio.

 "Closet Land" stars Alan Rickman and Madeleine Stowe.

 Ron Howard and Brian Grazer produced the feature.

 The screenplay for "Closet Land" won the Nicholl Screenwriting Fellowship sponsored by the Academy of Motion Picture Arts and Sciences, among others.
</item 3>

<item 4>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 4>

<item 5>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 5>

<item 6>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 6>

<item 7>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 7>

<item 8>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 8>

<item 9>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 9>

<item 10>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 10>
=== END ITEM id=e53bb15fa9da4345 ===

=== ITEM id=7f5c85d2ba1fd66c ===
Gold sentences and facts:
Sentence 0: Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the west.
  Fact 0.0: Longkou was formerly known as Huang County.
  Fact 0.1: Longkou is a port city.
  Fact 0.2: Longkou is in northeastern Shandong province, China.
  Fact 0.3: Longkou faces the Bohai Sea to the north.
  Fact 0.4: Longkou faces the Laizhou Bay to the west.
Sentence 1:  The district faces East China Sea and is connected to Mawei district by the Min River.
  Fact 1.0: The district faces the East China Sea.
  Fact 1.1: The district is connected to Mawei district by the Min River.

Retrieved items:
<item 1>
Xujiacao Changle Station
Xujiacao Changle Station () is a station on Line 1 of the Ningbo Rail Transit that started operations on 30May 2014.

 It is situated over Wangchun Road () and Xinyuan Road () in the Haishu District of Ningbo City, Zhejiang Province, eastern China.
</item 1>

<item 2>
Yandang Mountains
Yandang Mountains or Yandangshan (Chinese:  雁蕩山 ,  雁荡山 ,  "Yàndàng Shān",  "Wild Goose Pond Mountain(s)") refers, in the broad sense, to a coastal mountain
range in southeastern Zhejiang province in eastern China, covering much of the prefecture-level city of Wenzhou (from Pingyang County in the south to Yueqing
County in the northeast) and extending to the county-level city of Wenling in Taizhou prefecture.

 The mountain range is divided in two by the Oujiang River, the two parts being the North Yandang and South Yandang.

 More narrowly, Yandangshan is also used more narrowly to refer to Mount Yandang , a specific part of the North Yandang around an ancient caldera near a small
town of the same name (雁荡镇 , "Yàndàng Zhèn").

 The highest peaks of North Yandang are located here, and this is also the main tourist spot.
</item 2>

<item 3>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 3>

<item 4>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 4>

<item 5>
Line 6, Fuzhou Metro
Line 6 of the Fuzhou Metro () or Airport Line () runs from Cangshan District to Changle City.

 It starts at Exhibition Center Station and ends at Fuzhou Changle International Airport Station.

 The total length is 40.9 km.

 Line 6's color is pink.

 The construction is planned to begin by the end of 2016 and ends in 2021, from Exhibition Center to Fuzhou Changle International Airport.
</item 5>

<item 6>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 6>

<item 7>
Feng Xingxi
Feng Xingxi (馮行襲) (died July 31, 910), courtesy name Zhengchen (正臣), formally Prince Zhongjing of Changle (長樂忠敬王), was a warlord late in the Chinese dynasty
Tang Dynasty who later became a subject of the succeeding Later Liang.

 He was tall and strong and known as "Green Face Feng" for his green birthmark on his face.
</item 7>

<item 8>
Changle District
(, Foochow Romanized: Diòng-lŏ̤h) is a district located in eastern Fujian province, China.

 Administered by Fuzhou, it occupies a land area of 648 km2 and a sea area of 1327 km2 .

 Changle was established in the sixth year of Emperor Wu-De (623 AD) during the Tang Dynasty, and it became a county-level city on February 18, 1994.

 The district faces East China Sea and is connected to Mawei district by the Min River.

 Due to an increase in businesses, the province is now one of the richest province in China.

 The city was upgraded into a district on August 2017 by a government proposal.
</item 8>

<item 9>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 9>

<item 10>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 10>
=== END ITEM id=7f5c85d2ba1fd66c ===

=== ITEM id=eb282f0b0af4df8f ===
Gold sentences and facts:
Sentence 0: "Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".
  Fact 0.0: 'Empty Orchestra' is the fourth episode of the third series of Inside No. 9.
  Fact 0.1: Inside No. 9 is a British dark comedy anthology television programme.
Sentence 1:  Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.
  Fact 1.0: 'Empty Orchestra' was written by Steve Pemberton.
  Fact 1.1: 'Empty Orchestra' was written by Reece Shearsmith.
  Fact 1.2: 'Empty Orchestra' was directed by Guillem Morales.
  Fact 1.3: 'Empty Orchestra' first aired on 7 March 2017.
  Fact 1.4: 'Empty Orchestra' first aired on BBC 2.
Sentence 2:  The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).
  Fact 2.0: The episode 'Empty Orchestra' is set in a karaoke booth.
  Fact 2.1: The episode 'Empty Orchestra' follows a group of work colleagues.
  Fact 2.2: The work colleague Greg is shown as Shearsmith.
  Fact 2.3: The work colleague Fran is shown as Sarah Hadland.
  Fact 2.4: The work colleague Connie is shown as Tamzin Outhwaite.
  Fact 2.5: The work colleague Janet is shown as Emily Howlett.
  Fact 2.6: The work colleague Duane is shown as Javone Prince.
  Fact 2.7: The colleagues are celebrating the promotion of Roger.
  Fact 2.8: Roger is shown as Pemberton.
Sentence 3:  Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and television productions, including army series "Red Cap" and crime drama "New Tricks".
  Fact 3.0: The performer described in this sentence came to national notice playing Melanie Owen in the BBC One soap opera EastEnders from 1998 until 2002.
  Fact 3.1: The performer has since starred in theatre productions.
  Fact 3.2: The performer has since starred in television productions.
  Fact 3.3: The performer has starred in the army series Red Cap.
  Fact 3.4: The performer has starred in the crime drama New Tricks.

Retrieved items:
<item 1>
Tamzin Outhwaite
Tamzin Maria Outhwaite ( ; born 5 November 1970) is a British actress from London.

 Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and
television productions, including army series "Red Cap" and crime drama "New Tricks".
</item 1>

<item 2>
Empty Orchestra
"Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".

 Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.

 The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily
Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).

 Rebekah Hinds also stars.
</item 2>

<item 3>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 3>

<item 4>
Melanie Owen
Melanie Jane "Mel" Owen (also Healy and Beale) is a fictional character from the BBC soap opera "EastEnders", played by Tamzin Outhwaite.

 She appeared from 19 October 1998 to 12 April 2002 when Outhwaite opted to leave the role.

 Mel was introduced by executive producer Matthew Robinson and became a prominent character during her tenure in the soap.
</item 4>

<item 5>
Out of Control (2002 film)
Out of Control is a 2002 British television film written and directed for BBC by Dominic Savage.

 It stars Tamzin Outhwaite as Shelley Richards, an impoverished single mother whose son, Dean (Danny Young), is involved in a string of crimes that lead to his
incarceration in a young offenders institute (YOI).

 David Morrissey plays prison warden Mike, who tries to keep Dean out of trouble but has difficultly watching him all the time.

 After continuous bullying at the YOI, Dean is no longer able to cope and makes a fatal decision.

 It is the third in a loose trilogy of films by Savage about social deprivation, following "Nice Girl" (2000) and "When I Was Twelve" (2001).
</item 5>

<item 6>
Vital Signs (TV series)
Vital Signs is a British television drama series airing on ITV from 2006.

 It stars Tamzin Outhwaite as a supermarket check-out operator who decides to become a doctor.

 The series co-stars William Beck, Fraser Ayres, Eve Best, Claudie Blakley, Lucinda Dryzek, Beth Goddard, Alfie Hunter, Brooke Kinsella, Harry Lloyd, Peter Rnic
and Steven Waddington.

 The filming of the show is based in numerous London hospitals and medical schools; predominantly the show has been shot in St George's Hospital and Medical
School.
</item 6>

<item 7>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 7>

<item 8>
Paradox (TV series)
Paradox is a 2009 British science fiction police drama, starring Tamzin Outhwaite as Detective Inspector Rebecca Flint.

 Written by Lizzie Mickery and produced by Clerkenwell Films for the BBC, it was filmed and set in Manchester, England.
</item 8>

<item 9>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 9>

<item 10>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 10>
=== END ITEM id=eb282f0b0af4df8f ===

=== ITEM id=edda28aadacd61aa ===
Gold sentences and facts:
Sentence 0:  Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.
  Fact 0.0: The club will compete in the league.
  Fact 0.1: The club will also participate in the FA Cup.
  Fact 0.2: The club will also participate in the EFL Cup.
  Fact 0.3: The club will also participate in the EFL Trophy.
Sentence 1: The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual knockout football competition in men's domestic English football.
  Fact 1.0: The EFL Cup is historically referred to as the League Cup.
  Fact 1.1: The EFL Cup is colloquially referred to as simply the League Cup.
  Fact 1.2: The EFL Cup is currently known as the Carabao Cup for sponsorship reasons.
  Fact 1.3: The EFL Cup is an annual knockout football competition.
  Fact 1.4: The EFL Cup is a competition in men's domestic English football.

Retrieved items:
<item 1>
2007–08 Wigan Athletic F.C. season
The 2007–08 Wigan Athletic F.C. season was the club's 30th season in the Football League and their third season in the Premier League.
</item 1>

<item 2>
1978–79 Wigan Athletic F.C. season
The 1978–79 season was the 41st season in the history of Wigan Athletic F.C. and their first as a professional club in the Football League.

 After finishing 2nd in the Northern Premier League during the previous season, the club was nominated to apply for Football League status, and were elected
into the league to replace Southport.

 After a poor start, the club exceeded expectations in the league, and towards the end of the season, a second consecutive promotion seemed possible.

 The club eventually fell short, finishing the season in 6th place with a total of 55 points.
</item 2>

<item 3>
2017–18 Wigan Athletic F.C. season
The 2017–18 season is Wigan Athletic's 86th year in existence and their first back in League One, after being relegated the previous season.

 Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.

 Paul Cook was appointed as manager on 31 May 2017, signing a 3 year deal, following the departure of much of the previous coaching staff the day before.
</item 3>

<item 4>
EFL Cup
The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual
knockout football competition in men's domestic English football.

 Organised by the English Football League (EFL), it is open to any club within the top four levels of the English football league system – 92 clubs in total –
comprising the top level Premier League, and the three divisions of the English Football League's own league competition (Championship, League One and League
Two).
</item 4>

<item 5>
List of Wigan Borough F.C. players
Wigan Borough F.C. was an English football club based in the town of Wigan.

 The club was founded in 1920 and joined the Lancashire Combination.

 In 1921, Borough turned professional when their application was accepted to play in the inaugural season of the newly formed Football League Third Division
North.

 The team played in the Football League for ten seasons, with their most successful season coming in 1928–29, finishing fourth in the league and reaching the
third round of the FA Cup.

 Wigan Borough folded during the 1931–32 season due to financial problems, and League football did not return to the town until Wigan Athletic F.C. were elected
into the Football League in 1978.
</item 5>

<item 6>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 6>

<item 7>
Trinidad and Tobago League Cup
The Trinidad and Tobago League Cup, or commonly known as the First Citizens Cup for sponsorship reasons, is the league cup style football competition open for
Trinidad and Tobago teams competing in the country's TT Pro League.

 Similar to the FA Trophy, it is played on a knockout (single elimination) basis in September and October towards the beginning of each Pro League season.

 Unlike the FA Trophy, where 36 teams enter each season from the top three tiers of the Trinidad and Tobago football league system and the Secondary Schools
Football League, only teams from the TT Pro League compete in the league cup.

 The knockout tournament was inaugurated in 2000 and is currently sponsored by First Citizens Bank.

 Although the league cup is one of the three major domestic trophies attainable by Trinidad and Tobago league teams, it is perceived as a lower priority than
the league championship and the FA Trophy.

 The current theme is "Where Winners Reign", with TT$110,000 to the winner, TT$20,000 to the runners-up, semifinal winners receive TT$7,000, quaterfinal winners
receive TT$5,000 and Play-off round winners receive TT$3,000.
</item 7>

<item 8>
2006–07 Wigan Athletic F.C. season
The 2006–07 Wigan Athletic F.C. season was the club's 29th season in the Football League and their second season in the Premier League.
</item 8>

<item 9>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 9>

<item 10>
1979–80 Wigan Athletic F.C. season
The 1979–80 season was the 42nd season in the history of Wigan Athletic F.C. and their second as a professional club in the Football League.
</item 10>
=== END ITEM id=edda28aadacd61aa ===

=== ITEM id=168dfd56d4af9d41 ===
Gold sentences and facts:
Sentence 0: Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
  Fact 0.0: Rolf Martin Theodor Dahlgren was born on 7 July 1932.
  Fact 0.1: Rolf Martin Theodor Dahlgren died on 14 February 1987.
  Fact 0.2: Rolf Martin Theodor Dahlgren was a Swedish-Danish botanist.
  Fact 0.3: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen.
  Fact 0.4: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen from 1973 until his death.
Sentence 1:  The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.
  Fact 1.0: The university has 23,473 undergraduate students.
  Fact 1.1: The university has 17,398 postgraduate students.
  Fact 1.2: The university has 2,968 doctoral students.
  Fact 1.3: The university has over 9,000 employees.

Retrieved items:
<item 1>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 1>

<item 2>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 2>

<item 3>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 3>

<item 4>
Rolf Dahlgren
Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
</item 4>

<item 5>
Ebba Witt-Brattström
Ebba Witt-Brattström (born 1953) is a Swedish scholar in comparative literature.

 She is Professor of Literature and head of department at Södertörn University outside Stockholm, and a well-known feminist.

 In the 1970s she was a member of the feminist organisation Grupp 8, and in 2005 she was one of the founders of the feminist political organisation and party
Feministiskt Initiativ, although she later distanced herself from the organisation and criticized its alleged strong left-wing tendencies.
</item 5>

<item 6>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 6>

<item 7>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 7>

<item 8>
Theodor von Heuglin
Martin Theodor von Heuglin (20 March 1824, Hirschlanden, Württemberg – 5 November 1876), was a German explorer and ornithologist.
</item 8>

<item 9>
Gertrud Dahlgren
Gertrud Dahlgren (1931–2009) was a Swedish botanist.

 She was born at Klippan in Scania, and after graduating from university at Helsingborg went to the University of Lund for graduate studies.

 There she obtained her M.Sc.

 in chemistry and biology.

 She was married to fellow botanist Rolf Dahlgren (1932–1987), who was killed in a car accident.

 The couple had three children, Susanna, Helena and Fredrik.

 She died in December 2009 at the age of 78.
</item 9>

<item 10>
Rotman School of Management
The Joseph L. Rotman School of Management commonly known as the Rotman School of Management, the Rotman School or just Rotman, is the University of Toronto's
graduate business school, located in Downtown Toronto.

 The University of Toronto has been offering undergraduate courses in commerce and management since 1901, but the school was formally established in 1950 as the
Institute of Business Administration, which was then changed to the Faculty of Management Studies in 1972 and subsequently shortened to the Faculty of
Management in 1986.

 The school was renamed in 1997 after the late Joseph L. Rotman (1935–2015), its principal benefactor.
</item 10>
=== END ITEM id=168dfd56d4af9d41 ===

=== ITEM id=6d818cd7fdb2e830 ===
Gold sentences and facts:
Sentence 0:  He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.
  Fact 0.0: The person described in this sentence was the brother of Alfred Jodl.
  Fact 0.1: Alfred Jodl was Chief of the Operations Staff of the OKW.
Sentence 1: Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
  Fact 1.0: Alfred Josef Ferdinand Jodl lived from 10 May 1890 to 16 October 1946.
  Fact 1.1: Alfred Jodl was a German general.
  Fact 1.2: Alfred Jodl was a war criminal during World War II.
  Fact 1.3: Alfred Jodl served as Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).

Retrieved items:
<item 1>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 1>

<item 2>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 2>

<item 3>
Hasso von Wedel (general)
Hasso von Wedel (20 November 1898 – 3 January 1961) was a German general who commanded the Wehrmacht Propaganda Troops during World War II.

 He was directly subordinate to the head of OKW Operations Staff ("Wehrmachtführungsstab, WFSt.")

, General Alfred Jodl.

 Wedel's Propaganda Department had control over the propaganda units and served to mediate between them and the Reich Propaganda Ministry of Joseph Goebbels.
</item 3>

<item 4>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 4>

<item 5>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 5>

<item 6>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 6>

<item 7>
Algernon Willis
Admiral of the Fleet Sir Algernon Usborne Willis (17 May 1889 – 12 April 1976) was a Royal Navy officer.

 He served in the First World War and saw action at the Battle of Jutland in May 1916.

 He also served in the Second World War as Commander-in-Chief, South Atlantic in which capacity he led actions against German and Japanese raiding ships.

 He continued his war service as Flag Officer commanding 3rd Battle Squadron and Second in command of the Eastern Fleet and then as Flag Officer commanding
Force H, the force which covered North African Operations, the Allied invasion of Sicily in July 1943 and then the Allied invasion of Italy in September 1943.

 He spent the final years of the war as Commander-in-Chief, Levant, in which capacity he conducted naval operations in support of the Dodecanese Campaign, and
then as Second Sea Lord, in which capacity he arranged the manpower for the campaign in the Pacific Ocean against the Imperial Japanese Navy.

 After the war he served as Commander-in-Chief, Mediterranean Fleet, in which role he was faced with unrest in Mandatory Palestine, before he became
Commander-in-Chief, Portsmouth.
</item 7>

<item 8>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 8>

<item 9>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 9>

<item 10>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 10>
=== END ITEM id=6d818cd7fdb2e830 ===

=== ITEM id=12ff2a9ff8d7b524 ===
Gold sentences and facts:
Sentence 0: The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.
  Fact 0.0: The Bears and I is a 1974 American drama film.
  Fact 0.1: The Bears and I was directed by Bernard McEveety.
  Fact 0.2: The Bears and I was written by John Whedon.
Sentence 1:  The film was released on July 31, 1974, by Buena Vista Distribution.
  Fact 1.0: The Bears and I was released on July 31, 1974.
  Fact 1.1: The Bears and I was released by Buena Vista Distribution.
Sentence 2: Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as co-director.
  Fact 2.0: Oceans (French: Océans) is a 2009 French nature documentary film.
  Fact 2.1: Oceans was directed by Jacques Perrin.
  Fact 2.2: Oceans was produced by Jacques Perrin.
  Fact 2.3: Oceans was co-written by Jacques Perrin.
  Fact 2.4: Oceans was narrated by Jacques Perrin.
  Fact 2.5: Jacques Cluzaud was co-director of Oceans.

Retrieved items:
<item 1>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 1>

<item 2>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 2>

<item 3>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 3>

<item 4>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 4>

<item 5>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 5>

<item 6>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 6>

<item 7>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 7>

<item 8>
Disney's Adventures of the Gummi Bears
Disney's Adventures of the Gummi Bears is an American animated television series that first aired in the United States from 1985 to 1991.

 The series was the first animated production by Walt Disney Animation Television, and loosely inspired by the gummy bear candies; Disney CEO Michael Eisner was
struck with inspiration for the show when his son requested the candies one day.

 The series premiered on NBC on September 14, 1985, and aired there for four seasons.

 The series moved to ABC for one season from 1989 to 1990 (airing alongside "The New Adventures of Winnie the Pooh" as the "Gummi Bears-Winnie the Pooh Hour"),
and concluded on September 6, 1991 as part of the Disney Afternoon television syndication package.

 Of the series' 65 shows, 30 were double-features, consisting of two 11-minute cartoons, thereby bringing the series total to 94 distinct episodes overall.
</item 8>

<item 9>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 9>

<item 10>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 10>
=== END ITEM id=12ff2a9ff8d7b524 ===

=== ITEM id=8824069bcdd59e17 ===
Gold sentences and facts:
Sentence 0:  The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
  Fact 0.0: The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part.
  Fact 0.1: Sammy Hagar played the rhythm guitar part on a Gibson acoustic.
Sentence 1: OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist Sammy Hagar.
  Fact 1.0: OU812 is pronounced 'Oh You Ate One Too'.
  Fact 1.1: OU812 is the eighth studio album by American hard rock band Van Halen.
  Fact 1.2: OU812 was released in 1988.
  Fact 1.3: OU812 is the second Van Halen album to feature vocalist Sammy Hagar.

Retrieved items:
<item 1>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 1>

<item 2>
OU812
OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist
Sammy Hagar.

 Van Halen started work on the album in September 1987 and completed it in April 1988, just one month before its release.
</item 2>

<item 3>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 3>

<item 4>
Yeah! (Fuel song)
"Yeah!"

 is the first song released from Fuel's album "Puppet Strings".

 It is also the first song released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 4>

<item 5>
Keisuke Kuwata (album)
Keisuke Kuwata is the eponymous studio album recorded by Keisuke Kuwata, a frontman of the Japanese rock band Southern All Stars.

 It was released by Taishita label under the Victor Entertainment in July 1988, shortly after the 10th anniversary of the band.

 Aside from the 1982 live recording album "Kamon Yuzo and Victor Wheels Live" released under the pseudonym Yūzō Kamon, it was first Kuwata's solo effort.
</item 5>

<item 6>
OU812 Tour
The OU812 Tour was a concert tour by hard rock band Van Halen.

 It was the second tour to include dates in Japan, and was the second with Sammy Hagar as vocalist (who had settled into the role more by this point).

 It followed the Monsters of Rock Tour 1988, which had formed the first part of the promotion for the "OU812" album.
</item 6>

<item 7>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 7>

<item 8>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 8>

<item 9>
Finish What Ya Started
"Finish What Ya Started" is a song by Van Halen taken from their 1988 album "OU812".

 Despite the album being seemingly complete, Eddie Van Halen came up with the riff at 2 in the morning and went down to his then-neighbor Sammy Hagar to show
it.

 Hagar let Eddie in, and the two played guitars in his balcony until they had a completed song.

 Once Eddie left, Hagar decided to write the lyrics despite being late at night.

 The theme wound up being unfulfilled sex, summed up by Hagar as "blue balls.

 In the song, Eddie Van Halen recorded his guitar part on a Fender Stratocaster plugged direct into the studio mixing console.

 The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
</item 9>

<item 10>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 10>
=== END ITEM id=8824069bcdd59e17 ===

=== ITEM id=cb73081c2778beeb ===
Gold sentences and facts:
Sentence 0: The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.
  Fact 0.0: The Straight Story is a 1999 film.
  Fact 0.1: The Straight Story is internationally co-produced.
  Fact 0.2: The Straight Story is a biographical road drama film.
  Fact 0.3: The Straight Story was directed by David Lynch.
Sentence 1: Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.
  Fact 1.0: Frozen is a 2013 film.
  Fact 1.1: Frozen is American.
  Fact 1.2: Frozen is a 3D computer-animated film.
  Fact 1.3: Frozen is a musical fantasy film.
  Fact 1.4: Frozen was produced by Walt Disney Animation Studios.
  Fact 1.5: Frozen was released by Walt Disney Pictures.

Retrieved items:
<item 1>
The Straight Story
The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.

 The film was edited and produced by Mary Sweeney, Lynch's longtime partner and co-worker.

 She co-wrote the script with John E. Roach.

 The film is based on the true story of Alvin Straight's 1994 journey across Iowa and Wisconsin on a lawn mower.

 Alvin (Richard Farnsworth) is an elderly World War II veteran who lives with his daughter Rose (Sissy Spacek), a kind woman with an intellectual disability.

 When he hears that his estranged brother Lyle (Harry Dean Stanton) has suffered a stroke, Alvin makes up his mind to go visit him and hopefully make amends
before he dies.

 Because Alvin's legs and eyes are too impaired for him to receive a driving license, he hitches a trailer to his recently purchased thirty-year-old John Deere
110 Lawn Tractor, having a maximum speed of about 5 miles per hour, and sets off on the 240 mile journey from Laurens, Iowa to Mount Zion, Wisconsin.
</item 1>

<item 2>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 2>

<item 3>
Robert Lopez
Robert Lopez (born February 23, 1975) is an American songwriter of musicals, best known for co-creating "The Book of Mormon" and "Avenue Q", and for composing
the songs featured in the Disney animated film "Frozen".

 He is the youngest of only twelve people who have won an Emmy, a Grammy, an Oscar and a Tony Award, and the quickest (10 years) to win all four.
</item 3>

<item 4>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 4>

<item 5>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 5>

<item 6>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 6>

<item 7>
The Blizzard (1923 film)
The Blizzard is a 1923 Swedish drama film directed by Mauritz Stiller, starring Einar Hanson, Mary Johnson, Pauline Brunius and Hugo Björne.

 The film's original Swedish title is Gunnar Hedes saga, which means "The story of Gunnar Hede".

 The narrative revolves a student who tries to save his family's mansion which is facing bankruptcy.

 The film is loosely based on the Selma Lagerlöf novel "The Tale of a Manor".
</item 7>

<item 8>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 8>

<item 9>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 9>

<item 10>
Woman's film
The woman's film is a film genre which includes women-centered narratives, female protagonists and is designed to appeal to a female audience.

 Woman's films usually portray "women's concerns" such as problems revolving around domestic life, the family, motherhood, self-sacrifice, and romance.

 These films were produced from the silent era through the 1950s and early 1960s, but were most popular in the 1930s and 1940s, reaching their zenith during
World War II.

 Although Hollywood continued to make films characterized by some of the elements of the traditional woman's film in the second half of the 20th century, the
term itself disappeared in the 1960s.

 The work of directors George Cukor, Douglas Sirk, Max Ophüls, and Josef von Sternberg has been associated with the woman's film genre.

 Joan Crawford, Bette Davis, and Barbara Stanwyck were some of the genre's most prolific stars.
</item 10>
=== END ITEM id=cb73081c2778beeb ===

=== ITEM id=9394147bc31c9a15 ===
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
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 1>

<item 2>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 2>

<item 3>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 3>

<item 4>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 4>

<item 5>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 5>

<item 6>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 6>

<item 7>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
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
Qionghai
Qionghai () is one of the seven county-level cities of Hainan province, China.

 Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.

 Within this area is the main city, Qionghai City.

 It is located in the east of the island at the mouth of the Wanquan River, 86 km from the provincial capital of Haikou.

 It has an area of 1692 km2 and in 2010, it had a population of 483,217.
</item 9>

<item 10>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 10>
=== END ITEM id=9394147bc31c9a15 ===
