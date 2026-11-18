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

=== ITEM id=7d28b779047a6d35 ===
Gold sentences and facts:
Sentence 0: Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)
  Fact 0.0: Julia Compton Moore's dates are February 10, 1929 to April 18, 2004.
  Fact 0.1: Julia Compton Moore was the wife of Lieutenant General (Ret.).
Sentence 1:  Hal Moore, a United States Army officer.
  Fact 1.0: Hal Moore is a United States Army officer.
Sentence 2: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 2.0: Harold Gregory "Hal" Moore, Jr. lived from February 13, 1922 to February 10, 2017.
  Fact 2.1: Harold Gregory "Hal" Moore, Jr. was a United States Army lieutenant general.
  Fact 2.2: Harold Gregory "Hal" Moore, Jr. was an author.
Sentence 3:  He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point class (1945) to be promoted to brigadier general, major general, and lieutenant general.
  Fact 3.0: Hal Moore was a recipient of the Distinguished Service Cross.
  Fact 3.1: The Distinguished Service Cross is the U.S. military's second highest decoration for valor.
  Fact 3.2: Hal Moore was the first of his West Point class (1945) to be promoted to brigadier general.
  Fact 3.3: Hal Moore was the first of his West Point class (1945) to be promoted to major general.
  Fact 3.4: Hal Moore was the first of his West Point class (1945) to be promoted to lieutenant general.

Retrieved items:
<item 1>
Distinguished Service Cross (United States)
The Distinguished Service Cross is the second highest military award that can be given to a member of the United States Army (and previously, the United States
Army Air Forces and the United States Air Force), for extreme gallantry and risk of life in actual combat with an armed enemy force.

 Actions that merit the Distinguished Service Cross must be of such a high degree that they are above those required for all other U.S. combat decorations but
do not meet the criteria for the Medal of Honor.

 The Distinguished Service Cross is equivalent to the Navy Cross (Navy and Marine Corps), the Air Force Cross (Air Force), and the Coast Guard Cross (Coast
Guard).
</item 1>

<item 2>
Julia Compton Moore
Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)

 Hal Moore, a United States Army officer.

 Her efforts and complaints in the aftermath of the Battle of Ia Drang prompted the U.S. Army to set up survivor support networks and casualty notification
teams consisting of uniformed officers, which are still in use.
</item 2>

<item 3>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 3>

<item 4>
Wayne W. Lambert
Wayne W. Lambert (born 1936) was an Air force Brigadier General (United States).

 He attended the United States Military Academy at West Point, New York, after already having served a year in the US Army.

 Lambert graduated from the Academy in 1959 and was commissioned a 2nd Lieutenant in the USAF that same year.

 He trained as a bomber pilot, receiving his Aviator badge in 1960.

 He has logged over 5,800 flying hours with the US Air Force.

 A seasoned combat veteran, Lambert flew over 225 combat missions in South East Asia as a B-52 crew member.

 He was awarded the Distinguished Flying Cross (United States) and received 10 awards of the Air Medal for his service during the Vietnamese War.

 After serving in a succession of lesser command positions, Lambert was promoted to the rank of Brigadier General in 1983, assuming command of the former
Strategic Air Command’s 7th Air Division, headquartered then at Ramstein Air Base in West Germany, and overseeing SAC operations throughout the European
theater.

 The Brigadier General retired from the US Air Force in 1989.
</item 4>

<item 5>
Richard E. Cavazos
Richard Edward Cavazos (born January 31, 1929), a Korean War recipient of the Distinguished Service Cross as a first lieutenant, who advanced in rank to become
the United States Army's first Hispanic four-star general.

 During the Vietnam War, as a lieutenant colonel, Cavazos was awarded a second Distinguished Service Cross.

 In 1976, Cavazos became the first Mexican American to reach the rank of brigadier general in the U.S. Army.

 Cavazos served with great distinction for thirty-three years, with his final command as head of the U.S. Army Forces Command.
</item 5>

<item 6>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 6>

<item 7>
Nadja West
Nadja Y. West (born 1961) is a United States Army lieutenant general and the 44th U.S. Army Surgeon General and Commanding General of the U.S. Army Medical
Command.

 West is the first black Army Surgeon General, and was the first black female, active-duty, major general and the first black female major general in Army
Medicine.

 West is also the first Army black female lieutenant general.

 She is the highest ranking female to have graduated from the United States Military Academy.

 She received the Army Distinguished Service Medal, the Defense Superior Service Medal, the Legion of Merit, and others.
</item 7>

<item 8>
Hal Moore
Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.

 He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point
class (1945) to be promoted to brigadier general, major general, and lieutenant general.
</item 8>

<item 9>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 9>

<item 10>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 10>
=== END ITEM id=7d28b779047a6d35 ===

=== ITEM id=b049b06f1ca11960 ===
Gold sentences and facts:
Sentence 0: Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.
  Fact 0.0: Bernard Malamud was born on April 26, 1914.
  Fact 0.1: Bernard Malamud died on March 18, 1986.
  Fact 0.2: Bernard Malamud was an American novelist.
  Fact 0.3: Bernard Malamud was an American short story writer.
Sentence 1: Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.
  Fact 1.0: Lois McMaster Bujold was born on November 2, 1949.
  Fact 1.1: Lois McMaster Bujold is an American speculative fiction writer.

Retrieved items:
<item 1>
Shards of Honor
Shards of Honor is an English language science fiction novel by Lois McMaster Bujold, first published in June 1986.

 It is a part of the Vorkosigan Saga, and is the first full-length novel in publication order.

 "Shards of Honor" is paired with Bujold's 1991 "Barrayar" in the omnibus "Cordelia's Honor" (1996).
</item 1>

<item 2>
Horizon (novel)
Horizon is a fantasy novel by American writer Lois McMaster Bujold.

 It is the fourth in the tetralogy "The Sharing Knife".
</item 2>

<item 3>
Barrayar
Barrayar is a science fiction novel by Lois McMaster Bujold.

 It was first published as four installments in "Analog" in July–October 1991, and then published in book form by Baen Books in October 1991.

 "Barrayar" won both the Hugo Award for Best Novel and the Locus Award for Best Science Fiction Novel in 1992.

 It is a part of the Vorkosigan Saga, and is the seventh full-length novel of the series, in publication order.

 "Barrayar" is a direct sequel to Bujold's first novel, "Shards of Honor" (1986), and the two are paired in the 1996 omnibus "Cordelia's Honor".
</item 3>

<item 4>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 4>

<item 5>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 5>

<item 6>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 6>

<item 7>
Cryoburn
Cryoburn is a science fiction novel by Lois McMaster Bujold, first published in October 2010.

 Part of the Vorkosigan Saga, it was nominated for the Hugo Award for Best Novel in 2011, as Bujold's ninth Best Novel nomination.

 Also in 2011, it was one of the top five finishers in the poll for the Locus Award for Best Science Fiction Novel.
</item 7>

<item 8>
Gentleman Jole and the Red Queen
Gentleman Jole and the Red Queen is a science fiction novel by Lois McMaster Bujold.

 It is an installment in Bujold's Vorkosigan Saga.

 Bujold has described it as "not a war story.

 It is about grownups."
</item 8>

<item 9>
Passage (Bujold novel)
Passage is a novel by American writer Lois McMaster Bujold, published in 2008.

 It is the third in the tetralogy "The Sharing Knife".
</item 9>

<item 10>
Lois McMaster Bujold
Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.

 She is one of the most acclaimed writers in her field, having won the Hugo Award for best novel four times, matching Robert A. Heinlein's record, not counting
his Retro Hugo.

 Her novella "The Mountains of Mourning" won both the Hugo Award and Nebula Award.

 In the fantasy genre, "The Curse of Chalion" won the Mythopoeic Award for Adult Literature and was nominated for the 2002 World Fantasy Award for best novel,
and both her fourth Hugo Award and second Nebula Award were for "Paladin of Souls".

 In 2011 she was awarded the Skylark Award.

 In 2013 she was awarded the Forry Award.

 In 2017 she won a Hugo Award for Best Series, for the Vorkosigan Saga.
</item 10>
=== END ITEM id=b049b06f1ca11960 ===

=== ITEM id=a893e07624b1a398 ===
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
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 1>

<item 2>
Sapsali
The Sapsali (삽살이) is a shaggy Korean breed of dog.

 The word is followed in Korean by either "gae" (meaning "dog") or the suffix "ee"/"i", but is most commonly romanized as "Sapsaree".

 Traditionally, these dogs were believed to dispel ghosts and evil spirits.
</item 2>

<item 3>
Chiribaya Dog
The Chiribaya Dog (Spanish: "perro Chiribaya" ) or Peruvian shepherd dog ("perro pastor Peruano ") was a pre-Columbian breed of dog from the southwest of Peru,
identified by the 42 mummies discovered by anthropologist Sonia Guillén Oneglio in the Ilo District, Moquegua Region, on the south coast of Peru.

 It has been established that it was a llama herding dog.

 The dogs were not only an important part of the social structure of the ancient Peruvians, but they received special treatment after death as well.

 The dog variety has been referred to in various Spanish-language documentaries under different terms, such as "el perro pastor Chribaya" ('the Chiribaya
shepherd dog') and "pastor Peruano " ('Peruvian shepherd'), though the ancient Peruvians did not keep sheep.

 Its original name is unknown.
</item 3>

<item 4>
The Bulldog is a muscular, wrinkled English breed, ranked fourth among US purebreds in 2007. West Siberian Laikas are versatile barking hunting dogs whose
formal breeding resumed in 1946. Mediterranean warren hounds, including Cretan and Canary Island hounds, typically hunt using smell, sight, and hearing. Segugio
Italiano has short- and wire-haired varieties. Guatemala’s national Dogo is locally recognized; Cuba’s Bull Mastiff-type Dog
</item 4>

<item 5>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 5>

<item 6>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 6>

<item 7>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 7>

<item 8>
American Eskimo: German Spitz-derived companion/watchdog, renamed amid WWI prejudice; popular U.S. circus performer. English Springer: affectionate gun dog for
flushing/retrieving, also sniffer; recognized 1902, with show/working lines. German Spitz Klein: small German Spitz-type toy/utility dog. Picardy Spaniel: older
French hunting spaniel, weatherproof; now less popular, prone to ear infections. Welsh Springer: red-and-white hunting dog, loyal
</item 8>

<item 9>
Segugio Italiano
The Segugio Italiano ] is an Italian breed of dog of the scenthound family.

 It comes in both short-haired and wire-haired varieties.

 It is thought to be an ancient breed, descended in pre-Roman eras from progenitor scenthounds in ancient Egypt.

 In 2009 ENCI (the Italian Kennel Club) registered 4,500 specimens of the short-haired variety and 1,740 wire-haired specimens, making this one of the top ten
breeds in Italy.
</item 9>

<item 10>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 10>
=== END ITEM id=a893e07624b1a398 ===

=== ITEM id=adc48af906cc6d72 ===
Gold sentences and facts:
Sentence 0:  The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
  Fact 0.0: The film concentrated on Ellsworth "Bumpy" Johnson.
  Fact 0.1: Laurence Fishburne played Ellsworth "Bumpy" Johnson in the film.
  Fact 0.2: The film concentrated on Dutch Schultz.
  Fact 0.3: Tim Roth played Dutch Schultz in the film.
  Fact 0.4: The film concentrated on Lucky Luciano.
  Fact 0.5: Andy García played Lucky Luciano in the film.
Sentence 1:  Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.
  Fact 1.0: Lucky Luciano is considered the father of modern organized crime in the United States.
  Fact 1.1: Lucky Luciano is considered the father of modern organized crime in the United States for establishing the first Commission.
  Fact 1.2: Lucky Luciano established the first Commission.

Retrieved items:
<item 1>
Frank Colacurcio
Francis Colacurcio, Sr. (June 18, 1917 – July 2, 2010) was an American businessman and boss of the Seattle crime family known for running strip clubs in
Seattle, Washington.

 He gained notoriety as a subject of ongoing federal investigations into organized crime in the city and was suspected of being an organized crime boss.
</item 1>

<item 2>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 2>

<item 3>
James Marcello
James J. Marcello (born December 13, 1943), also known variously as Little Jimmy, Jimmy Light and as Jimmy the Man Marcello, is an imprisoned crime boss who was
a front boss for the Chicago Outfit criminal organization in the 1980s, 1990s and early 2000s.

 Organized crime observers identified Marcello as a figurehead during that period while the organization's day-to-day operations actually were run by John “No
Nose” DiFronzo, Joseph “Joey the Clown” Lombardo, Joseph “Joe the Builder” Andriacchi and Angelo J. LaPietra.

 Marcello is the Cousin of Michael Hayes Marcello, and a cousin of Robert Urbinatti.

 He is also the father of Rocco Marcello.
</item 3>

<item 4>
Hoodlum (film)
Hoodlum is a 1997 American crime drama film that gives a fictionalized account of the gang war between the Italian/Jewish mafia alliance and the Black gangsters
of Harlem that took place in the late 1920s and early 1930s.

 The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
</item 4>

<item 5>
The Wolf of Wall Street (2013 film)
The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir
of the same name by Jordan Belfort.

 It recounts Belfort's perspective on his career as a stockbroker in New York City and how his firm Stratton Oakmont engaged in rampant corruption and fraud on
Wall Street that ultimately led to his downfall.

 Leonardo DiCaprio (who was also a producer) stars as Belfort, with Jonah Hill as his business partner and friend Donnie Azoff, Margot Robbie as his wife Naomi
Lapaglia and Kyle Chandler as Patrick Denham, the FBI agent who tries to bring him down.

 Matthew McConaughey, Rob Reiner, Jon Favreau, Joanna Lumley and Jean Dujardin also star.
</item 5>

<item 6>
Hoodlum (soundtrack)
Hoodlum is the soundtrack to the 1997 American crime drama film "Hoodlum".

 It was released on August 12, 1997 by Interscope Records and consisted of a blend of hip hop and R&B music.

 The soundtrack peaked at 94 on the "Billboard" 200 and 23 on the Top R&B Albums and contained Mobb Deep's single "Hoodlum" which went to 29 on the Hot Rap
Singles.
</item 6>

<item 7>
Joseph Todaro Jr.
Joseph "Big Joe" Todaro Jr. (born 1945 or 1946) is a Buffalo, New York businessman and former organized crime figure involved in labor racketeering,
loansharking, illegal gambling, narcotics, and murder for hire.

 Joe Todaro Jr. became a business agent for the Laborers' International Union of North America (LIUNA) Local 210.

 In 1984, Joe Todaro Jr. allegedly became the underboss of the Buffalo crime family after his father became the new head of the crime family, upon the
retirement of his predecessor Samuel "Sam the Farmer" Frangiamore.

 In 1990, Joe Todaro Jr. resigned as business agent following investigations on the local's alleged ties to organized crime.

 Outside of organized crime, Todaro operates La Nova Pizzeria, a popular pizza restaurant in Buffalo.
</item 7>

<item 8>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 8>

<item 9>
The Yakuza
The Yakuza is a 1974 Japanese-American neo-noir gangster film directed by Sydney Pollack, written by Leonard Schrader, Paul Schrader, and Robert Towne.

 The film is about a man (Robert Mitchum) who returns to Japan after several years away in order to rescue his friend's kidnapped daughter.

 Following a lackluster initial release, the film has since gained a cult following.
</item 9>

<item 10>
Ang Utol Kong Hoodlum
Ang Utol Kong Hoodlum (lit.

 "My Brother is a Hoodlum") is a Filipino drama series developed for TV5 created by Deo J. Fajardo.

 It stars JC de Vera and Jasmine Curtis-Smith.

 It is a remake of the original movie where Robin Padilla played the role of Ben, a hoodlum, and Vina Morales, as Bing.

 The movie was first released in 1991, then a sequel entitled "Miss na Miss Kita: Ang Utol Kong Hoodlum 2" was made the following year.

 It is produced by Vic Del Rosario Jr., and Manuel V. Pangilinan and it also marks as the first primetime series produced by Viva Television for TV5 after a
decade.
</item 10>
=== END ITEM id=adc48af906cc6d72 ===

=== ITEM id=58db4239b9fa1235 ===
Gold sentences and facts:
Sentence 0: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 0.0: Henry Louis Mencken lived from September 12, 1880 to January 29, 1956.
  Fact 0.1: Henry Louis Mencken was an American journalist.
  Fact 0.2: Henry Louis Mencken was an American satirist.
  Fact 0.3: Henry Louis Mencken was an American cultural critic.
  Fact 0.4: Henry Louis Mencken was a scholar of American English.
Sentence 1: Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.
  Fact 1.0: Albert Camus lived from 7 November 1913 to 4 January 1960.
  Fact 1.1: Albert Camus was a French philosopher.
  Fact 1.2: Albert Camus was an author.
  Fact 1.3: Albert Camus was a journalist.

Retrieved items:
<item 1>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 1>

<item 2>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 2>

<item 3>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 3>

<item 4>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 4>

<item 5>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 5>

<item 6>
Reflections on the Guillotine
"Reflections on the Guillotine" is an extended essay written in 1957 by Albert Camus.

 In the essay Camus takes an uncompromising position for the abolition of the death penalty.

 Camus's view is similar to that of Cesare Beccaria and the Marquis de Sade, the latter having also argued that murder premeditated and carried out by the state
was the worst kind.

 Camus states that he does not base his argument on sympathy for the convicted but on logical grounds and on proven statistics.

 Camus also argues that capital punishment is an easy option for the government where remedy and reform may be possible.
</item 6>

<item 7>
The Guest
"The Guest" (French: "L'Hôte" ) is a short story by the French writer Albert Camus.

 It was first published in 1957 as part of a collection entitled "Exile and the Kingdom" ("L'exil et le royaume").

 The French title "L'Hôte" translates into both "the guest" and "the host" which ties back to the relationship between the main characters of the story.

 Camus employs this short tale to reflect upon issues raised by the political situation in French North Africa.

 In particular, he explores the problem of refusing to take sides in the colonial conflict in Algeria, something that mirrors Camus' own non-aligned stance
which he had set out in his Nobel Prize acceptance speech.
</item 7>

<item 8>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 8>

<item 9>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 9>

<item 10>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 10>
=== END ITEM id=58db4239b9fa1235 ===

=== ITEM id=b54b09308a3c9a3c ===
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
Daryl Palumbo
Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.

 He is the frontman of the bands Glassjaw, Head Automatica and Color Film.

 As a youth he was a member of the Long Island straight edge band "XbustedX."

 In 1993 he met guitarist Justin Beck, who would become his good friend.

 Together they formed Glassjaw.
</item 1>

<item 2>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 2>

<item 3>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 3>

<item 4>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 4>

<item 5>
Phil Driscoll
Phil Driscoll (born November 9, 1947) is a trumpeter, singer, composer, and producer.

 He performs in varying music genres and styles which include rock and roll, rhythm and blues, and patriotic music, and is best known for his work in Christian
music and his longterm Christian ministry.

 In 1985, Driscoll won the Grammy Award for Best Gospel Performance – Duo/Group for a duet with Debby Boone on "Keep the Flame Burning", and he has been
nominated for three additional Grammys, two for Best Gospel Performance – Male and one for Best Gospel/Pop Album.

 He has also won three Dove Awards for his music, and the 1999 Christian Country Music Association Award for Best Musician.

 In 2006, Driscoll was found guilty on 2 counts of tax evasion and one count of conspiracy, and was sentenced to serve one year in Federal prison, beginning on
March 14, 2007.
</item 5>

<item 6>
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
</item 6>

<item 7>
Head Automatica
Head Automatica is an American rock band from Long Island, New York fronted by Daryl Palumbo (also of Glassjaw).
</item 7>

<item 8>
Tom Johnston
Tom Johnston, Thomas Johnston or Tommy Johnston may refer to:
</item 8>

<item 9>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 9>

<item 10>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 10>
=== END ITEM id=b54b09308a3c9a3c ===

=== ITEM id=b2d423d69113fa42 ===
Gold sentences and facts:
Sentence 0: Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.
  Fact 0.0: Terri Kathleen Nunn was born June 26, 1961.
  Fact 0.1: Terri Kathleen Nunn is an American singer.
  Fact 0.2: Terri Kathleen Nunn is an American actress.
Sentence 1: Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.0: Sonya Scarlet was born 2 April 1980.
  Fact 1.1: Sonya Scarlet is the singer of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.2: Sonya Scarlet is the lyricist of the Italian extreme gothic metal band Theatres des Vampires.

Retrieved items:
<item 1>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 1>

<item 2>
Sonya Yoncheva
Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
</item 2>

<item 3>
Norine Burgess
Norine Burgess is a Canadian singer.

 She is a graduate of the University of Calgary and the University of Toronto’s Opera School, mezzo-soprano.

 She received additional training as a member of the Canadian Opera Company (COC) Ensemble where she appeared in Electra, Suor Angelica, Lulu and Der
Rosenkavalier.

 Ms. Burgess also performed Le Nozze di Figaro(Cherubino), Ariadne auf Naxos (Dryad), La Traviata(Flora) and Die Zauberflöte (Second Lady) with the COC.

 Additional operatic successes include Le Nozze di Figaro (Cherubino) and Carmen (Mercedes) with the Vancouver Opera, Fenena in Nabucco with Manitoba Opera,
Albert Herring (Nancy) with the Calgary Opera and many appearances with the Edmonton Opera.
</item 3>

<item 4>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 4>

<item 5>
Sonya Scarlet
Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
</item 5>

<item 6>
Terri Nunn
Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.

 She is best known as the lead vocalist of the new wave/synthpop band Berlin.
</item 6>

<item 7>
Briana Corrigan
Briana Corrigan (born 30 May 1965) is a Northern Irish singer.

 She was the first female singer for The Beautiful South, from 1988 to 1992.
</item 7>

<item 8>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 8>

<item 9>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 9>

<item 10>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 10>
=== END ITEM id=b2d423d69113fa42 ===

=== ITEM id=6f0effa024b1e494 ===
Gold sentences and facts:
Sentence 0:  Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any affiliation with the work.
  Fact 0.0: The film takes its title from an album by Leonard Cohen.
  Fact 0.1: Ira Nadel wrote a biography of Leonard Cohen.
  Fact 0.2: The subject of the film is not Leonard Cohen.
  Fact 0.3: The subject of the film has no affiliation with the work.
Sentence 1: Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and painter.
  Fact 1.0: Leonard Norman Cohen was born on September 21, 1934.
  Fact 1.1: Leonard Norman Cohen died on November 7, 2016.
  Fact 1.2: Leonard Norman Cohen was a Canadian singer.
  Fact 1.3: Leonard Norman Cohen was a Canadian songwriter.
  Fact 1.4: Leonard Norman Cohen was a Canadian musician.
  Fact 1.5: Leonard Norman Cohen was a Canadian poet.
  Fact 1.6: Leonard Norman Cohen was a Canadian novelist.
  Fact 1.7: Leonard Norman Cohen was a Canadian painter.

Retrieved items:
<item 1>
Daddy's Hands (band)
Daddy's Hands was a Canadian experimental rock band formed in Victoria, British Columbia, Canada, by singer/guitarist Dave Wenger, bassist Emily Bauslaugh,
drummer Jonah Fortune, and saxophonist Jonathan Pollard.

 Dave Wenger and Emily Bauslaugh had been involved in the British Columbian hardcore/emo scene before forming the band, being in bands M Blanket and Ache Hour
Credo, and the band Floragore respectively.

 Daddy's Hands has released two demo tapes, one EP and two full-length albums.

 The band gigged extensively in the 90s, even playing a radio show with Nardwuar.

 Despite achieving little commercial success, the band has been referred to as highly influential in the West Coast music scene.

 Lead singer Wenger, highly regarded by some critics, has been referred to as Montreal's Peter Laughner.
</item 1>

<item 2>
Various Positions (film)
Various Positions is a 2002 film by Vancouver, BC, lawyer and filmmaker Ori Kowarsky, starring Carly Pope and Tygh Runyan.

 "Various Positions" won the 2002 Prix de Montréal at the Montreal World Film Festival.

 Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any
affiliation with the work.
</item 2>

<item 3>
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
</item 3>

<item 4>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 4>

<item 5>
For Those Who Think Young (album)
For Those Who Think Young, appearing on the album cover as (for those who think young) and originally to be entitled for those who think jung, was the third
album by Canadian new wave band Rough Trade; it was released in 1981 (True North TN-48 in Canada; Boardwalk NB-33261-1, US; Big Time, UK; CBS 85385, The
Netherlands).

 It climbed to #9 in Canada on the "RPM" Top 50 Albums Chart on November 7, 1981 (putting it at #1 on the CANCON Chart listing), and held the position for three
weeks, dropping out of the Top 50 after sixteen weeks on February 6 of the following year.

 It was certified gold in Canada by the CRIA on November 1, 1981.

 The single "All Touch" gave the band its biggest commercial success, reaching #12 in Canada on the "RPM" Top 50 Singles Chart (#2 on the CANCON Chart) and #58
on the U.S. ""Billboard" Hot 100".
</item 5>

<item 6>
Various Positions Tour
The Various Positions Tour was a concert tour by Leonard Cohen, in support of his album "Various Positions", released in 1984 (Canada) and 1985 in Europe and
the United States.
</item 6>

<item 7>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 7>

<item 8>
Love Can't Wait (Nick Carter song)
"Love Can't Wait" is a song by American singer-songwriter Nick Carter, released as the lead Canadian single from his second studio album, "I'm Taking Off".

 The single was released on June 21, 2011.

 The song only appears on the Canadian version of the album, and has not been released in any other territories.

 The music video for the song features Carter performing the song with the backdrop of a motorway system.

 The music video also featured Canadian pop star Shawn Desman, who co-wrote the song.

 The single peaked at #2 on the Canadian Singles Chart, based entirely on download sales.

 Carter ran a competition for amateur mixers to remix the track, with the winning remixer earning $1500 and their remix appearing on Carter's remix album, "I'm
Taking Off: Relaunched and Remixed".
</item 8>

<item 9>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 9>

<item 10>
New World Man
"New World Man" is a track from the 1982 album "Signals" by Canadian rock band Rush.

 The song was the last and quickest composed song on the album, stemming from a suggestion by then-Rush producer Terry Brown to even out the lengths of the two
sides of the cassette version.

 It went to #1 (on the "RPM" national singles chart) in Canada, where it remained for two weeks in October 1982.

 Less successful in the United States, it nonetheless remains Rush's only American Top 40 hit, peaking at #21 on the "Billboard" singles chart for three weeks
in October and November 1982.

 It also topped the "Billboard" Top Tracks chart for two weeks (their first single to do so).

 "New World Man" also reached #42 in the UK; a remixed version released as a double A-side with "Countdown" later reached #36 in the UK in early 1983.
</item 10>
=== END ITEM id=6f0effa024b1e494 ===

=== ITEM id=2afeae8edcc21664 ===
Gold sentences and facts:
Sentence 0: Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central business district of Brainerd, a city in Crow Wing County, Minnesota, United States.
  Fact 0.0: Brainerd Lakes Regional Airport has IATA code BRD.
  Fact 0.1: Brainerd Lakes Regional Airport has ICAO code KBRD.
  Fact 0.2: Brainerd Lakes Regional Airport has FAA LID BRD.
  Fact 0.3: Brainerd Lakes Regional Airport is a public use airport.
  Fact 0.4: Brainerd Lakes Regional Airport is located three nautical miles (6 km) northeast of the central business district of Brainerd.
  Fact 0.5: Brainerd is a city in Crow Wing County, Minnesota, United States.
Sentence 1: Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City, Utah in the United States.
  Fact 1.0: Salt Lake City International Airport has IATA code SLC.
  Fact 1.1: Salt Lake City International Airport has ICAO code KSLC.
  Fact 1.2: Salt Lake City International Airport has FAA LID SLC.
  Fact 1.3: Salt Lake City International Airport is a civil-military airport.
  Fact 1.4: Salt Lake City International Airport is located about 4 miles west of Downtown Salt Lake City.
  Fact 1.5: Downtown Salt Lake City is in Utah, United States.

Retrieved items:
<item 1>
900 South (UTA station)
900 South is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by all three lines of Utah Transit Authority's TRAX light rail
system.

 The Blue Line provides service from Downtown Salt Lake City to Draper.

 The Red Line provides service from the University of Utah to the Daybreak Community of South Jordan.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).

 The station opened on 19 September 2005 and is operated by the Utah Transit Authority.

 900 South is notable for being the first infill station constructed along an existing line of the TRAX system.
</item 1>

<item 2>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 2>

<item 3>
Arena (UTA station)
Arena is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit Authority's
(UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line has service from the Salt Lake City International Airport and to West Valley City via Downtown Salt Lake City.
</item 3>

<item 4>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 4>

<item 5>
Courthouse (UTA station)
Courthouse is a light rail station in Downtown Salt Lake City, Utah, United States serviced by all three lines of Utah Transit Authority's TRAX light rail
system.

 The Blue Line provides service from Downtown Salt Lake City to Draper.

 The Red Line provides service from the University of Utah to the Daybreak Community of South Jordan.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 5>

<item 6>
Temple Square (UTA station)
Temple Square is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 6>

<item 7>
Brainerd Lakes Regional Airport is a city- and county-owned, general-aviation airport near Brainerd, Minnesota, with one commercial airline. Grand Sky,
established in 2015 at Grand Forks Air Force Base, is a 217-acre UAS business park with runway access. Grand Forks Municipal Airport closed after World War II
and became urban land; Grand Forks International Airport serves the area and offers customs but no scheduled international flights. Nearby Emerado had 414
residents in
</item 7>

<item 8>
Salt Lake City Air Route Traffic Control Center
Salt Lake City Air Route Traffic Control Center (ZLC) (radio communications "Salt Lake Center") is one of 22 FAA Area Control Centers in the United States.

 It is located in Salt Lake City, Utah, adjacent to Salt Lake City International Airport.

 It was opened in 1939 and was originally located on the third floor of the old Salt Lake City International Airport terminal.

 The Salt Lake Center (ZLC) covers one of the largest geographical areas of any other control center, totaling approximately 350,000 squares miles.
</item 8>

<item 9>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 9>

<item 10>
City Center (UTA station)
City Center is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 10>
=== END ITEM id=2afeae8edcc21664 ===

=== ITEM id=e679276799817b6c ===
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
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 1>

<item 2>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 2>

<item 3>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 3>

<item 4>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 4>

<item 5>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 5>

<item 6>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 6>

<item 7>
Foxes (soundtrack)
Foxes is the soundtrack to the 1980 film of the same name, starring Jodie Foster, Scott Baio, Sally Kellerman, Randy Quaid as well as The Runaways' lead singer
Cherie Currie.

 The double-album was released on the disco label Casablanca Records.
</item 7>

<item 8>
Return to the Blue Lagoon
Return to the Blue Lagoon is a 1991 American romance and adventure film starring Milla Jovovich and Brian Krause, produced and directed by William A. Graham.

 The film is a sequel to "The Blue Lagoon" (1980).

 The screenplay by Leslie Stevens was based on the novel "The Garden of God" by Henry De Vere Stacpoole.

 The original music score was composed by Basil Poledouris.

 The film's closing theme song "A World of Our Own" is performed by Surface featuring Bernard Jackson.

 The music was written by Barry Mann, and the lyrics were written by Cynthia Weil.
</item 8>

<item 9>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 9>

<item 10>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 10>
=== END ITEM id=e679276799817b6c ===

=== ITEM id=cdd522776a4483e2 ===
Gold sentences and facts:
Sentence 0: Robert Harry Philibosian (born 1940) is an American politician.
  Fact 0.0: Robert Harry Philibosian was born in 1940.
  Fact 0.1: Robert Harry Philibosian is an American politician.
Sentence 1: David R. Ignatius (May 26, 1950), is an American journalist and novelist.
  Fact 1.0: David R. Ignatius was born on May 26, 1950.
  Fact 1.1: David R. Ignatius is an American journalist.
  Fact 1.2: David R. Ignatius is an American novelist.

Retrieved items:
<item 1>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 1>

<item 2>
Andrew Benjamin
Andrew Benjamin (born 1952, Australia) is an Australian philosopher.

 Currently he holds two academic posts: Distinguished Professor of Philosophy and the Humanities at Kingston University, London and Professor of Philosophy and
Jewish Thought in the Department of Philosophy and the Australian Centre for Jewish Civilization at Monash University, Melbourne, Australia.

 Benjamin first came to critical attention with his writings in continental philosophy, writing articles and editing books on the thinking of Martin Heidegger,
Theodor Adorno, Walter Benjamin, Julia Kristeva and Jean-François Lyotard.

 Benjamin has become involved in the field of architecture, to the extent that he has also taught in various schools of architecture in UK, US and Australia.
</item 2>

<item 3>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 3>

<item 4>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 4>

<item 5>
Robert Philibosian
Robert Harry Philibosian (born 1940) is an American politician.

 He served as Los Angeles County District Attorney from 1982 to 1984.

 He received his B.A. in history from Stanford University and his law degree from Southwestern Law School, and was admitted to the California State Bar in 1968.

 He is now Of counsel at the law firm of Sheppard, Mullin, Richter & Hampton.
</item 5>

<item 6>
Mark Krikorian
Mark Krikorian has been the executive director of the Center for Immigration Studies, a think-tank in Washington, D. C., since 1995.

 The Center describes itself as an "independent, non-partisan research organization" in Washington, D. C., that examines and critiques the impact of immigration
on the United States.

 Animated by a "pro-immigrant, low-immigration vision which seeks fewer immigrants, but a warmer welcome for those admitted", the Center was established in 1985
to provide immigration research.

 Krikorian is a regular contributor to the conservative publication "National Review", and is a regular participant at "National Review Online's" "The Corner."
</item 6>

<item 7>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
</item 7>

<item 8>
Richard Ziser
Richard Ziser is a Nevada Real Estate Investor, Socially Conservative Political activist and U.S. Republican Politician.

 He was born June 7, 1953, in Pomona, CA., and has resided in Las Vegas Nevada since 1991.

 He graduated from California State Polytechnic University, Pomona (Cal Poly Pomona) with a BS in Industrial Engineering, 1976; then subsequently from Simon
Greenleaf University in Santa Ana, Ca.

 (now a campus of Trinity International University, with an MA in Christian Apologetics in 1989.
</item 8>

<item 9>
David I. Walsh
David Ignatius Walsh (November 11, 1872June 11, 1947) was a United States politician from Massachusetts.

 A member of the Democratic Party, he served as the 46th Governor of Massachusetts before serving several terms in the United States Senate.
</item 9>

<item 10>
Phil Driscoll
Phil Driscoll (born November 9, 1947) is a trumpeter, singer, composer, and producer.

 He performs in varying music genres and styles which include rock and roll, rhythm and blues, and patriotic music, and is best known for his work in Christian
music and his longterm Christian ministry.

 In 1985, Driscoll won the Grammy Award for Best Gospel Performance – Duo/Group for a duet with Debby Boone on "Keep the Flame Burning", and he has been
nominated for three additional Grammys, two for Best Gospel Performance – Male and one for Best Gospel/Pop Album.

 He has also won three Dove Awards for his music, and the 1999 Christian Country Music Association Award for Best Musician.

 In 2006, Driscoll was found guilty on 2 counts of tax evasion and one count of conspiracy, and was sentenced to serve one year in Federal prison, beginning on
March 14, 2007.
</item 10>
=== END ITEM id=cdd522776a4483e2 ===
