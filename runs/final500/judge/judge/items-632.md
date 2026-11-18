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

=== ITEM id=ea5d6cd235ec7b99 ===
Gold sentences and facts:
Sentence 0:  In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
  Fact 0.0: In 2003, a person referred to as "he" received an award for his actions during the Battle of Qala-i-Jangi.
  Fact 0.1: The Battle of Qala-i-Jangi took place in late November to early December 2001.
Sentence 1: The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November 25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
  Fact 1.0: The Battle of Qala-i-Jangi is also incorrectly referred to as the "Battle of Mazar-i-Sharif".
  Fact 1.1: The Battle of Qala-i-Jangi was a prisoner-of-war camp uprising.
  Fact 1.2: The Battle of Qala-i-Jangi took place between November 25 and December 1, 2001.
  Fact 1.3: The Battle of Qala-i-Jangi took place in northern Afghanistan.
  Fact 1.4: The Battle of Qala-i-Jangi followed the armed intervention by United States-led coalition forces.
  Fact 1.5: The United States-led coalition forces intervened with armed forces to try to overthrow the Taliban's Islamic Emirate of Afghanistan.
  Fact 1.6: The coalition forces had accused the Taliban's Islamic Emirate of Afghanistan of harboring al-Qaeda operatives.

Retrieved items:
<item 1>
Iraultza
Iraultza (Basque for Revolution) was a small Basque militant armed group of leftist tendency.

 Seven of the group would die preparing explosive devices.

 On June 27, 1986, a bomb claimed by Iraultza mangled a young worker called José Miguel Moros Peña in Portugalete.

 The victim would die in Cruces Hospital (Barakaldo) on August 13, 1986.

 The group is also known for being described by the United States Government as "probably [having] committed more bombings against American business interests
than any other European terrorist group".

 Soon after, the group, which had been active for about a decade with small attacks of explosives that never aimed to cause any personal injuries, announced its
dissolution.

 When the group was operational, it is estimated that it had approximately 20 members.
</item 1>

<item 2>
Enemy Combatant (book)
Enemy Combatant is a memoir by British Muslim, Moazzam Begg, co-written by Victoria Brittain, former Associate Foreign Editor for "The Guardian", about Begg's
detention by the government of the United States of America in Bagram Detention Facility and at Camp Echo, Guantanamo Bay and his life prior to that detention.

 It was published in Britain as "Enemy Combatant: A British Muslim's Journey To Guantanamo and Back" (ISBN  ), and in the US as "Enemy Combatant: My
Imprisonment at Guantanamo, Bagram, and Kandahar" (ISBN  ).

 In the US, the foreword was written by David Ignatius of The Washington Post.
</item 2>

<item 3>
Battle of Qala-i-Jangi
The Battle of Qala-i-Jangi (also incorrectly referred to as the "Battle of Mazar-i-Sharif") was a prisoner-of-war camp uprising that took place between November
25 and December 1, 2001, in northern Afghanistan, following the armed intervention by United States-led coalition forces to try to overthrow the Taliban's
Islamic Emirate of Afghanistan, which it had accused of harboring al-Qaeda operatives.
</item 3>

<item 4>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 4>

<item 5>
The Great War for Civilisation
The Great War for Civilisation: The Conquest of the Middle East is a book published in 2005 by the award-winning English journalist Robert Fisk.

 The book is a compilation of many of the articles Fisk wrote when he was serving as a correspondent in the Middle East for "The Times" and "The Independent".

 The book revolves around several key themes regarding the history of the modern Middle East: the Arab-Israeli conflict, the Soviet invasion of Afghanistan, the
Persian Gulf War as well as the 2003 Iraq War (United States invasion of Iraq) as well as other regional conflicts such as the Armenian Genocide and the
Algerian Civil War.

 The "Great War for Civilisation" is the second book Fisk has written about the Middle East with the first one, "Pity the Nation", (Nation Books, 2002) being
about the Lebanese Civil War.
</item 5>

<item 6>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 6>

<item 7>
Orlov let
Orlov let ("Eagle's Flight") was the name of an anti-terrorist operation conducted by the Montenegrin police to arrest a group of Albanians who planned
terrorist attacks and an armed conflict in Albanian-inhabited parts of Montenegro.

 The group of 17 people planned operations from 2004 to their arrest on the night of 10/11 September 2006.

 Weapons and explosive material were found in police raids.

 The group's main operative was the destroying of cultural and religious buildings (Orthodox) in Albanian-inhabited territory.

 The group was organised by an Albanian association based in Detroit, and members of the Kosovo Liberation Army, from which they received financial aid and
smuggled arms into Montenegro.

 The group was given a total sentence of 51 years of prison.

 Relations between the state and the Albanian minority strained after the event.
</item 7>

<item 8>
Combat Terrorist Organization
Combat Terrorist Organization or Autonomous Combat Terrorist Organization (rus.

 Боевая Террористическая Организация) was a Russian radical Neo-Nazi terrorist group based in Saint Petersburg from 9 August 2003 to 2006.

 It was formed by two members of the "Mad Crowd" skinhead organization, Dmitriy Borovikov and Voevodin Alexey.
</item 8>

<item 9>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 9>

<item 10>
Sayeed Salahudeen
Syed Mohammed Yusuf Shah, popularly known as Syed Salahudeen, is the head of Hizb-ul-Mujahideen, a terrorist group operating in Kashmir, and head of an alliance
of anti-India militant groups, the United Jihad Council, that works to annex the Indian state of Jammu and Kashmir to Pakistan .

 Salahuddin vowed to block any peaceful resolution to the Kashmir conflict, threatened to train more Kashmiri suicide bombers, and vowed to turn the Kashmir
valley “into a graveyard for Indian forces.”

 He is listed on the NIA Most Wanted list.

 On 26 June 2017, he was named as a Specially Designated Global Terrorist by the US Department Of State.

 After being designated as a Global terrorist, he addressed the Pakistani Media at Muzaffarabad’s Centre Press Club, claiming that "The declaration was a joint
move by the US, Israel, and India to express their animosity towards Pakistan."
</item 10>
=== END ITEM id=ea5d6cd235ec7b99 ===

=== ITEM id=aa7a18f199ee46a0 ===
Gold sentences and facts:
Sentence 0:  In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.
  Fact 0.0: In 2010, the person referred to as "he" began providing vocals in the pop rock band Ashland HIGH.
  Fact 0.1: In 2010, the person referred to as "he" began providing guitar in the pop rock band Ashland HIGH.
Sentence 1: Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.
  Fact 1.0: Jizzy Pearl was born James Wilkinson.
  Fact 1.1: Jizzy Pearl is an American vocalist.
  Fact 1.2: Jizzy Pearl was born on March 17, 1958.
  Fact 1.3: Jizzy Pearl was born in Chicago, Illinois.

Retrieved items:
<item 1>
Jizzy Pearl
Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.

 He's had notable performances in several heavy metal and hard rock groups.

 He first fronted the band Data Clan, which eventually became Love/Hate.

 Pearl has also sung for L.A. Guns, Ratt, Adler's Appetite, and other, lesser known acts.
</item 1>

<item 2>
Trace Cyrus
Trace Dempsey Cyrus (born Neil Timothy Helson; February 24, 1989) is an American musician.

 The son of country singer Billy Ray Cyrus and brother of recording artist Miley Cyrus, he is the backing vocalist and bassist of the band Metro Station.

 He left the band in 2010 but returned in 2014.

 In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.

 He also owns the clothing company From Backseats to Bedrooms.
</item 2>

<item 3>
Brother Clyde (album)
Brother Clyde is the debut studio album from alternative rock group Brother Clyde.

 It is the first side-project from country music singer Billy Ray Cyrus.

 The album features duets with Cyrus' son Trace Cyrus as well as Dolly Parton, who Cyrus said "wanted to rock."

 It was released on August 10, 2010.

 Its first single, "Lately", which features rapper King Phaze, was issued to digital retailers on June 29, 2010, and a music video, which was directed by Cyrus,
was released for the song in July 2010.
</item 3>

<item 4>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 4>

<item 5>
Shrinking Violet (album)
Shrinking Violet is the sixth album by glam metal band L.A. Guns, first released on June 1, 1999, through Perris Records, and is the only L.A. Guns album with
singer Jizzy Pearl.

 The album was reissued, with bonus tracks and new artwork, on May 24, 2010, through Favored Nations.

 The album was produced by former Guns N' Roses guitarist Gilby Clarke who also provided additional guitars on the track "Dreamtime".
</item 5>

<item 6>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 6>

<item 7>
Ron Cyrus
Ronald Ray "Ron" Cyrus (July 10, 1935 – February 28, 2006) was an American Democratic politician and public servant in Greenup County, Kentucky.

 He was the father of American country music singer/actor Billy Ray Cyrus and the paternal grandfather of Trace Cyrus, Miley Cyrus and Noah Cyrus.
</item 7>

<item 8>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 8>

<item 9>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 9>

<item 10>
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 10>
=== END ITEM id=aa7a18f199ee46a0 ===

=== ITEM id=7faa3660fafb10cf ===
Gold sentences and facts:
Sentence 0:  The event was broadcast live on BBC One and was hosted by Graham Norton.
  Fact 0.0: The event was broadcast live on BBC One.
  Fact 0.1: Graham Norton hosted the event.
Sentence 1: Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.
  Fact 1.0: Graham William Walker was born on 4 April 1963.
  Fact 1.1: Graham William Walker is better known as Graham Norton.
  Fact 1.2: Graham Norton is an Irish television presenter.
  Fact 1.3: Graham Norton is an Irish radio presenter.
  Fact 1.4: Graham Norton is an Irish DJ.
  Fact 1.5: Graham Norton is an Irish comedian.
  Fact 1.6: Graham Norton is an Irish actor.
  Fact 1.7: Graham Norton is an Irish writer.

Retrieved items:
<item 1>
British Academy Television Awards
The British Academy Television Awards, also known as the BAFTA Television Awards, are presented in an annual award show hosted by the British Academy of Film
and Television Arts (BAFTA).

 They have been awarded annually since 1955.

 It is the British equivalent to the Emmy Awards in the United States and Logie Awards in Australia.
</item 1>

<item 2>
British Academy Television Awards 1998
The 1998 British Academy Television Awards were held on 18 May at the Prince of Wales Theatre in London.

 The ceremony was broadcast on ITV, hosted by Bob Monkhouse, and it was the first occasion since 1968 that the Television Awards had been held separately from
the British Academy Film Awards, instead of as a joint ceremony.
</item 2>

<item 3>
BAFTA Scotland
BAFTA in Scotland is the Scottish branch of the British Academy of Film and Television Arts.

 Formed in 1986, the branch holds two annual awards ceremonies recognising the achievement by performers and production staff in Scottish film, television and
video games.

 These Awards are separate from the British Academy Television Awards and British Academy Film Awards.
</item 3>

<item 4>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 4>

<item 5>
British Academy Children's Awards
The British Academy Children's Awards are presented in an annual award show hosted by the British Academy of Film and Television Arts (BAFTA).

 They have been awarded annually since 1996.

 Before that, children's awards were a part of the main British Academy Television Awards.
</item 5>

<item 6>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 6>

<item 7>
British Academy Television Award for Best Actress
This is a list of the British Academy Television Awards for Best Actress.

 The British Academy Television Awards began in 1955.

 The Best Actress award was initially given as an 'individual honour' without credit to a particular performance until 1969 when Wendy Craig won for her
performance in "Not in Front of the Children".

 Since 1970, nominees have been announced in addition to the winner and are listed with the winner highlighted in blue.

 The Actress category was split into Leading Actress and Supporting Actress starting in 2010.
</item 7>

<item 8>
British Academy Television Awards 1997
The 1997 British Academy Television Awards were held on 29 April at the Royal Albert Hall in London, as a joint ceremony with the British Academy Film Awards.

 To date, it is the last occasion upon which the two sets awards have been given jointly at the same event; from 1998, the Film and Television Awards were given
at separate ceremonies, for the first time since 1968.
</item 8>

<item 9>
2013 Africa Magic Viewers Choice Awards
The 2013 Africa Magic Viewers Choice Awards were held on 9 March 2014 and co-hosted by Big Brother Africa presenter IK Osakioduwa and StarGist’s host, Vimbai
Mutinhiri.

 Ivie Okujaiye and Olu Jacobs were honored with the TrailBlazer and Industry Merit Awards respectively.
</item 9>

<item 10>
Graham Norton
Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.

 He is a five-time BAFTA TV Award winner for his comedy chat show "The Graham Norton Show".

 Previously shown on BBC Two, it took the prestigious Friday night slot on BBC One from "The Jonathan Ross Show" in 2010.
</item 10>
=== END ITEM id=7faa3660fafb10cf ===

=== ITEM id=e56729387a03b932 ===
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
Segugio Italiano
The Segugio Italiano ] is an Italian breed of dog of the scenthound family.

 It comes in both short-haired and wire-haired varieties.

 It is thought to be an ancient breed, descended in pre-Roman eras from progenitor scenthounds in ancient Egypt.

 In 2009 ENCI (the Italian Kennel Club) registered 4,500 specimens of the short-haired variety and 1,740 wire-haired specimens, making this one of the top ten
breeds in Italy.
</item 1>

<item 2>
Chiribaya Dog
The Chiribaya Dog (Spanish: "perro Chiribaya" ) or Peruvian shepherd dog ("perro pastor Peruano ") was a pre-Columbian breed of dog from the southwest of Peru,
identified by the 42 mummies discovered by anthropologist Sonia Guillén Oneglio in the Ilo District, Moquegua Region, on the south coast of Peru.

 It has been established that it was a llama herding dog.

 The dogs were not only an important part of the social structure of the ancient Peruvians, but they received special treatment after death as well.

 The dog variety has been referred to in various Spanish-language documentaries under different terms, such as "el perro pastor Chribaya" ('the Chiribaya
shepherd dog') and "pastor Peruano " ('Peruvian shepherd'), though the ancient Peruvians did not keep sheep.

 Its original name is unknown.
</item 2>

<item 3>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 3>

<item 4>
Santa's Little Helper
Santa's Little Helper is a recurring character in the American animated television series "The Simpsons".

 He is the pet greyhound of the Simpson family.

 The dog was introduced in the first episode of the show, the 1989 Christmas special "Simpsons Roasting on an Open Fire", in which his owner abandons him for
finishing last in a greyhound race.

 Homer Simpson and his son Bart, who are at the race track in hope of winning some money for Christmas presents, see this and decide to adopt the dog.
</item 4>

<item 5>
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 5>

<item 6>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 6>

<item 7>
Sapsali
The Sapsali (삽살이) is a shaggy Korean breed of dog.

 The word is followed in Korean by either "gae" (meaning "dog") or the suffix "ee"/"i", but is most commonly romanized as "Sapsaree".

 Traditionally, these dogs were believed to dispel ghosts and evil spirits.
</item 7>

<item 8>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 8>

<item 9>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 9>

<item 10>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 10>
=== END ITEM id=e56729387a03b932 ===

=== ITEM id=f5e45a8389ca2156 ===
Gold sentences and facts:
Sentence 0: The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.
  Fact 0.0: The Celluloid Closet is a 1995 American documentary film.
  Fact 0.1: The Celluloid Closet was directed by Rob Epstein and Jeffrey Friedman.
  Fact 0.2: The Celluloid Closet was written by Rob Epstein and Jeffrey Friedman.
Sentence 1: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.
  Fact 1.0: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening".
  Fact 1.1: The film Station to Station is made up of 62 one-minute films.

Retrieved items:
<item 1>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 1>

<item 2>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 2>

<item 3>
Hard to Explain
"Hard to Explain" is the first single from New York garage rock band The Strokes (their only previously released material was "The Modern Age" EP).

 It was released first in the UK and was later released in the US with different album artwork.

 (The UK version has a photo of two chairs, one red and one black, facing the camera.

 The chairs appear to be in a diner or restaurant of some sort.)

 Because this single is the first from their debut LP "Is This It", "Hard to Explain" made the anticipation for the album proper very high, and when "Is This
It" did come out it was widely hailed as one of the best of the year.

 The B-side of this single, "New York City Cops" was omitted from the US version of the album in the wake of the September 11 attacks on the World Trade Center
(the chorus to the song contains the lines "New York City cops/They ain't too smart").
</item 3>

<item 4>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 4>

<item 5>
Romance of Celluloid
The Romance of Celluloid is a 1937 short black and white documentary film, narrated by Frank Whitbeck, which goes behind the scenes to look at the manufacture
of film and the making of motion pictures.

 The film was the first of the studio's "Romance of Celluloid" series, which also included;
</item 5>

<item 6>
Station to Station (film)
Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.

 It was distributed internationally in 2015 following its release at the Sundance Film Festival.
</item 6>

<item 7>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 7>

<item 8>
The Boxer (Kele Okereke album)
The Boxer is the debut solo album by Kele Okereke, the lead singer of British indie rock band Bloc Party.

 Okereke released the album under the professional name of Kele on 21 June 2010.

 As promotion, he uploaded the songs "Rise" and "Walk Tall" to his personal website on 13 May.

 The first single from "The Boxer" was "Tenderoni", released on 14 June, and the second, "Everything You Wanted", on 16 August.

 The album's third single "On The Lam" was released on 25 October.
</item 8>

<item 9>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 9>

<item 10>
Metro Station (band)
Metro Station is an American pop rock band that was formed in Los Angeles by Trace Cyrus and Mason Musso.

 In late 2006, the band signed a recording contract with Columbia Records and RED Ink Records.

 The band is best known for the Top 10 "Billboard" hit single "Shake It" from the group's self-titled debut album.

 In 2010, tension between Cyrus and Musso caused the band to go on hiatus.

 In 2011 the band returned, however it was announced that Cyrus was no longer a part of the group and Musso had purchased the rights to the name.

 An EP entitled "Middle of the Night" was released in 2013, which was led by the single "Every Time I Touch You".

 In 2014 Cyrus and a new drummer, Spencer Steffan came to the band and a new single entitled "Love & War" was released.

 In 2015, the band released a second full-length album titled "Savior".
</item 10>
=== END ITEM id=f5e45a8389ca2156 ===

=== ITEM id=6f59a267bff47c8d ===
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
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 1>

<item 2>
Review
A review is an evaluation of a publication, service, or company such as a movie (a movie review), video game (video game review), musical composition (music
review of a composition or recording), book (book review); a piece of hardware like a car, home appliance, or computer; or an event or performance, such as a
live music concert, play, musical theater show, dance show, or art exhibition.

 In addition to a critical evaluation, the review's author may assign the work a rating to indicate its relative merit.

 More loosely, an author may review current events, trends, or items in the news.

 A compilation of reviews may itself be called a review. "

The New York Review of Books", for instance, is a collection of essays on literature, culture, and current affairs. "

National Review", founded by William F. Buckley, Jr., is an influential conservative magazine, and "Monthly Review" is a long-running socialist periodical.
</item 2>

<item 3>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 3>

<item 4>
Charles C. W. Cooke
Charles C. W. Cooke (born November 4, 1984) is the editor of "National Review Online".

 He took the role over after Rich Lowry stepped down in June, 2016 (Lowry remains the editor-in-chief of "National Review").

 Cooke is the author of "The Conservatarian Manifesto" and a frequent guest on HBO's "Real Time with Bill Maher".

 In addition to "National Review", he has written for the "New York Times", the "Washington Post", and the "Los Angeles Times".

 Along with Kevin D. Williamson, he hosts the popular "Mad Dogs and Englishmen" podcast.
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
National Review Board
The National Review Board (full name: National Review Board for the Protection of Children and Young People) is a committee created in 2002 by the United States
Conference of Catholic Bishops in order to monitor the implementation of the Charter for the Protection of Children and Young People in the wake of the clerical
abuse scandal in the United States.

 The Board was also charged with investigating the scandal, which it did in part by commissioning the John Jay College to conduct a survey of Church records in
order to define the nature and scope of the abuse committed by priests over the period between 1950 and 2002.

 The results of that survey were released in 2004 in what has come to be known as the "John Jay Report" or "Nature and Scope report."

 In parallel with the John Jay College survey the Board conducted interviews with a variety of people, both inside and outside the Church, who were well placed
to comment on the scandal, and on the basis of these interviews prepared a more broad-ranging report of its own.
</item 7>

<item 8>
The Human Life Review
The Human Life Review is a quarterly journal published by the Human Life Foundation since 1975.

 It is devoted to explorations of life issues, primarily abortion, as well as neonaticide, medical genetics, prenatal testing, human cloning, fetal tissue
experimentation, euthanasia and assisted suicide, and also publishes articles dealing with more general questions of family and society.

 It was founded by James Patrick McFadden, formerly associate publisher of "National Review", who had also founded the Human Life Foundation, and is now edited
by his daughter, Maria McFadden.

 It was launched from the offices of "National Review", with the support of William F. Buckley.
</item 8>

<item 9>
Kevin D. Williamson
Kevin Daniel Williamson (born September 18, 1972) is the roving correspondent for "National Review."

 He is also the theater critic for "The New Criterion".

 He was previously deputy managing editor at "National Review".
</item 9>

<item 10>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 10>
=== END ITEM id=6f59a267bff47c8d ===

=== ITEM id=320a3edbb7f9a8b0 ===
Gold sentences and facts:
Sentence 0: Dana Robins Ivey (born August 12, 1941) is an American actress.
  Fact 0.0: Dana Robins Ivey was born on August 12, 1941.
  Fact 0.1: Dana Robins Ivey is an American actress.
Sentence 1: Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.
  Fact 1.0: Two Weeks Notice is a 2002 film.
  Fact 1.1: Two Weeks Notice is an Australian-American film.
  Fact 1.2: Two Weeks Notice is a romantic comedy film.
  Fact 1.3: Two Weeks Notice stars Hugh Grant.
  Fact 1.4: Two Weeks Notice stars Sandra Bullock.

Retrieved items:
<item 1>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 1>

<item 2>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 2>

<item 3>
Dana Ivey
Dana Robins Ivey (born August 12, 1941) is an American actress.

 She is a five-time Tony Award nominee for her work on Broadway, and won the 1997 Drama Desk Award for Outstanding Featured Actress in a Play for her work in
both "Sex and Longing" and "The Last Night of Ballyhoo".

 Her film appearances include "The Color Purple" (1985), "Dirty Rotten Scoundrels" (1988), "The Addams Family" (1991), "Two Weeks Notice" (2002), "Rush Hour 3"
(2007) and "The Help" (2011).
</item 3>

<item 4>
Thank God It's Friday (film)
Thank God It's Friday is a 1978 American musical comedy film directed by Robert Klane and produced by Motown Productions and Casablanca FilmWorks for Columbia
Pictures (whose torch-holding mascot, in a specially produced animation, dances to disco music before the opening credits).

 Produced at the height of the disco craze, the film features The Commodores performing "Too Hot ta Trot", and Donna Summer performing "Last Dance", which won
the Academy Award for Best Song in 1978.

 The film features an early performance by Jeff Goldblum and the first major screen appearance by Debra Winger.

 The film also features Terri Nunn who would go on to fame in the 1980s new wave group Berlin.
</item 4>

<item 5>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 5>

<item 6>
Two Weeks (The Office)
"Two Weeks" is the twenty-first episode of the fifth season of the television series "The Office", and the 93rd overall episode of the series.

 It originally aired on NBC in the United States on March 26, 2009.

 In this episode, Michael, who has given his two weeks' notice to Dunder Mifflin, tries to convince others in the office to quit and join him in starting a new
paper company.

 Meanwhile, Pam spends her day trying to put together the new photocopier and becomes frustrated with her job.
</item 6>

<item 7>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 7>

<item 8>
Dana (surname)
Dana ( or ) as a surname may have several origins.

 In England, it came from "dann", the valley of a meadow, and it may mean the dweller of that valley.

 In Continental Europe, it probably came from "Dane" (or Danish, from Denmark).

 This surname is related to Danese in Italy and it can be found mostly in the Piedmont region.

 It may also be a modification of Huguenot French origin, probably a variant of "d'Aunay", of geographical origin.

 It may be also a Gaelic patronymic, since it is a common forename in Ireland.

 Dana is a relatively common surname in the US, ranking 7161 out of 88,799 in the 1990 U.S. Census.

 Dana is also the Persian word for wisdom.
</item 8>

<item 9>
Two Weeks Notice
Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.

 The film was written and directed by Marc Lawrence, who had previously scripted two other box-office hits for Bullock, "Miss Congeniality" and "Forces of
Nature".

 Although critical response was mixed, the film received a successful box office run, both in the United States and globally.
</item 9>

<item 10>
Return to the Blue Lagoon
Return to the Blue Lagoon is a 1991 American romance and adventure film starring Milla Jovovich and Brian Krause, produced and directed by William A. Graham.

 The film is a sequel to "The Blue Lagoon" (1980).

 The screenplay by Leslie Stevens was based on the novel "The Garden of God" by Henry De Vere Stacpoole.

 The original music score was composed by Basil Poledouris.

 The film's closing theme song "A World of Our Own" is performed by Surface featuring Bernard Jackson.

 The music was written by Barry Mann, and the lyrics were written by Cynthia Weil.
</item 10>
=== END ITEM id=320a3edbb7f9a8b0 ===

=== ITEM id=d40f93dc488940ab ===
Gold sentences and facts:
Sentence 0: In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
  Fact 0.0: In the middle of 1984, a Brazilian company called Prológica brought a new equipment to the Brazilian market
  Fact 0.1: Prológica made its own versions of 8 bits US computers
  Fact 0.2: The new equipment was for Prológica's personal computer series
  Fact 0.3: The new equipment was called "CP"
  Fact 0.4: CP is short for Personal Computer in Portuguese
Sentence 1: The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.
  Fact 1.0: The Atari ST is a line of home computers from Atari Corporation
  Fact 1.1: The Atari ST is the successor to the Atari 8-bit family
Sentence 2:  The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.
  Fact 2.0: The Atari ST is the first personal computer to come with a bitmapped color GUI
  Fact 2.1: The Atari ST's bitmapped color GUI uses a version of Digital Research's GEM
  Fact 2.2: The version of Digital Research's GEM used by the Atari ST was released in February 1985

Retrieved items:
<item 1>
Megamax C
Megamax C is a K&R implementation of C for the Atari ST, Apple IIgs, and Macintosh computers.

 Sold by Megamax, Inc., based in Richardson, Texas, the package includes a one-pass compiler, linker, editor, resource construction kit, and documentation.

 In the early days of the Atari ST, Megamax C was the primary competitor to the Alcyon C compiler from Digital Research which was included in the official
developer kit from Atari.

 According to a review in "Antic" magazine, Megamax C compiled a small benchmark program six times faster than Digital Research's compiler.
</item 1>

<item 2>
DTACK Grounded
DTACK Grounded was a computer hobbyist newsletter published from July 1981 to September 1985 by Hal W. Hardenberg.

 Subtitled "The Journal of Simple 68000 Systems", the newsletter was dedicated to the proposition that the Motorola 68000 CPU could be used to build simple fast
computers.

 In 1981 this was a revolutionary idea.

 This was before 68000-based personal computers like the Apple Macintosh, Commodore Amiga, and Atari ST had been created.

 In 1981 Motorola was marketing the 68000 solely as a CPU for high-end Unix workstations.

 Hal believed that the 68000 could be used as a simple embedded microprocessor as well, and used the newsletter to explain how to do that.
</item 2>

<item 3>
Page 6
Page 6 (subtitled "Atari Users Magazine", and later known as New Atari User) was an independent British publication aimed at users of Atari home computers.

 It was published between 1982 and 1998.

 The computer magazine supported both the Atari 8-bit family of computers (400/800/XL/XE) and the Atari ST range.
</item 3>

<item 4>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 4>

<item 5>
ATASCII
The ATASCII character set, from ATARI Standard Code for Information Interchange", alternatively ATARI ASCII", is the variation on ASCII used in the Atari 8-bit
family of home computers.

 The first of this family were the Atari 400 and 800, released in 1979, and later models were released throughout the 1980s.

 The last computer to use the ATASCII character set was the Atari XEGS which was released in 1987.

 The Atari ST family of computers used the different Atari ST character set.
</item 5>

<item 6>
XaAES
FreeMint, MultiTOS, and a few more obscure OSes are all successors to TOS the proprietary operating system of the Atari ST computer.

 Later models of the ST Computer were called the TT and the Falcon.

 TOS stands for The Operating System or Tramiel Operating System, as Jack Tramiel was the owner of Atari.

 GEMDOS itself was a near clone of DOS, the IBM PC OS.

 This was most visible to users in the 8.3 character file naming system.

 Like all personal computer operating systems of the day, TOS was single tasking.

 GEM, the graphical user interface, was licensed from Digital Research and was not included in Apple's lawsuit against DR, thus the Macintosh like appearance
and ways of doing things remained on Atari computers.

 As time went on, the big goal for the ST within and without Atari was to have a multitasking TOS.
</item 6>

<item 7>
Happy drives
Happy drives are series of disk drive enhancements for the Atari 8-bit and Atari ST computer families produced by a small company Happy Computers.

 Happy Computers is most noted for the add-in boards for the Atari 810 and Atari 1050 disk drives, which achieved a tremendous speed improvement for reading and
writing, and for the ability to "back up" floppies.

 Happy's products were among the most popular Atari computer add-ons.

 They were still in use and active in the aftermarket as of 2009.
</item 7>

<item 8>
Prológica CP-400
In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment
for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
</item 8>

<item 9>
Atari ST
The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.

 The initial ST model, the 520ST, saw limited release in April-June 1985 and was widely available in July.

 The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.

 The 1040ST, released in 1986, is the first personal computer to ship with a megabyte of RAM in the base configuration and also the first with a
cost-per-kilobyte of less than US$1.
</item 9>

<item 10>
Atari Corporation
Atari Corporation was an American manufacturer of computers and video game consoles from 1984 to 1996.

 Atari Corp. was founded in July 1984 when Warner Communications sold the home computing and game console divisions of Atari, Inc. to Jack Tramiel.

 Its chief products were the Atari ST, , Atari 7800, Atari Lynx, and Atari Jaguar.

 The company reverse merged with JTS Inc. in 1996, becoming a small division, which itself closed when JTS liquidated the IP to Hasbro Interactive in 1998.
</item 10>
=== END ITEM id=d40f93dc488940ab ===

=== ITEM id=9b322b373e6f8b0e ===
Gold sentences and facts:
Sentence 0: Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.
  Fact 0.0: Ash Lieb was born on 22 August 1982.
  Fact 0.1: Ash Lieb is an Australian artist.
  Fact 0.2: Ash Lieb is an Australian writer.
  Fact 0.3: Ash Lieb is an Australian comedian.
  Fact 0.4: Ash Lieb is known for his surreal humour.
  Fact 0.5: Ash Lieb is known for his surreal art.
Sentence 1: Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.
  Fact 1.0: Helen Elswyth Thane Ricker Beebe was born on May 16, 1900.
  Fact 1.1: Helen Elswyth Thane Ricker Beebe died on July 31, 1984.
  Fact 1.2: Helen Elswyth Thane Ricker Beebe was an American romance novelist.

Retrieved items:
<item 1>
Ash Lieb
Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.

 Born in Ballarat, Ash Lieb began exhibiting art at eight years of age, and at the age of fifteen, wrote his first novel, "The Secret Well".

 Throughout his career, Lieb has created a diverse range of artworks, books, short films, and comedic performances, which have often possessed philosophical or
psychiatric undertones.
</item 1>

<item 2>
Lev Yilmaz
Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic
series.
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
Tryst (novel)
Tryst, written in 1939 by Elswyth Thane, is a story of two people and a seemingly ordinary home.

 While a quick summary may make it sound like a Horror novel, it actually borders on Mystery and Romance.
</item 4>

<item 5>
Lorin Thompson
Lorin Thompson, Lorin Hartwell Jr Thompson, (1911–1997), was a muralist, artist, and creator and illustrator of the character "Ranger Rick" for the National
Wildlife Federation's children's magazine, Ranger Rick.
</item 5>

<item 6>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 6>

<item 7>
Dorian Gray (character)
Dorian Gray is the main character in Oscar Wilde's novel The Picture of Dorian Gray.

 Dorian Gray is a handsome, susceptible, wealthy young Victorian gentleman.

 He falls in love with and devotes himself to Sibyl Vane.

 Dorian has two friends, Basil and Henry.

 Basil is an artist who draws a portrait of Dorian that emphasizes his youthful, noble, and natural beauty.

 Henry is a friend who has a cynical nature and who shares various thoughts with Dorian.
</item 7>

<item 8>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 8>

<item 9>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 9>

<item 10>
Elswyth Thane
Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.

 Born in Burlington, Iowa, she was the daughter of a local teacher and high school principal.

 The family moved to New York City in 1918, and "Helen Ricker" changed her name to "Elswyth Thane".

 She began working as a freelance writer in the 20s, and became a newspaper writer and a Hollywood screenwriter.

 Her first novel, "Riders of the Wind", was published in 1926.

 Her novel, "The Tudor Wench", about Elizabeth I of England, was made into a play.

 She was a collector of scarves.
</item 10>
=== END ITEM id=9b322b373e6f8b0e ===

=== ITEM id=883cf2f7bdd590e7 ===
Gold sentences and facts:
Sentence 0: Ionolyce is a small genus of butterflies in the family Lycaenidae.
  Fact 0.0: Ionolyce is a small genus of butterflies.
  Fact 0.1: Ionolyce is in the family Lycaenidae.
Sentence 1:  "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.
  Fact 1.0: I. helicon is widespread in the Indoaustralasian region.
  Fact 1.1: I. helicon is widespread from India to the Malay Archipelago.
Sentence 2: India lies on the Indian Plate, the northern portion of the Indo-Australian Plate, whose continental crust forms the Indian subcontinent.
  Fact 2.0: India lies on the Indian Plate.
  Fact 2.1: The Indian Plate is the northern portion of the Indo-Australian Plate.
  Fact 2.2: The continental crust of the Indian Plate forms the Indian subcontinent.
Sentence 3:  It is the seventh-largest country in the world, with a total area of 3287263 km2 .
  Fact 3.0: India is the seventh-largest country in the world.
  Fact 3.1: India has a total area of 3287263 km2.

Retrieved items:
<item 1>
Ionolyce
Ionolyce is a small genus of butterflies in the family Lycaenidae.

 "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.

 "I selkon"and "I.brunnescens" are both endemic to the Solomon Islands.
</item 1>

<item 2>
The entries cover diverse taxa and concepts: butterflies (Mormon metalmark, tiger leafwing, Indian/Malayan leafwing, and Ionolyce blues), moths (three Aulacodes
species and the pale western cutworm), birds (ground jays, honeyguides, black terns), and other organisms. They note habitats, ranges, diets, camouflage, and
ecological roles. Also described are sand tampan ticks, midge orchids, Cretaceous herbivorous
</item 2>

<item 3>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 3>

<item 4>
An encyclopedia-style collection covers deer and other wildlife, noting habitats, ranges, traits, and conservation status: Sri Lankan sambar are exceptionally
large; moose are solitary northern browsers; several Asian deer are threatened, while Indian muntjac is least concern. It also mentions the Calypso
fish-identification database, wild boar, and black bull. Other entries describe the Umba River and Babati; Ovambo, Oku, and Ewe peoples; Hawaiian muumuus
</item 4>

<item 5>
The material surveys wildlife and ecological traits, including butterflies, moths, birds, ticks, orchids, and Cretaceous herbivores; restaurant chains spanning
global fast food and regional cuisines; and hot sauces from Valentina and Tabasco to Puerto Rican pique. It also covers apple dumplings, mulling spices, Finnish
Christmas traditions, and cocktails such as Flaming Volcano and Blue Lagoon, noting the spectacle of flaming drinks.
</item 5>

<item 6>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 6>

<item 7>
An encyclopedia-style collection spans plant biology, ecology, wildlife, food, and culture. It covers herbicides (imazapyr and HPPD inhibitors), plant defenses
and diversity, Huernia’s fly-attracting flowers, Oenanthe javanica, and Nepenthes pitcher plants, especially in Southeast Asia. Wildlife entries include
butterflies, birds, ticks, and deer, from large Sri Lankan sambar to threatened Asian species. Other topics include global and regional restaurants, hot sauces
</item 7>

<item 8>
A botanical survey covers African Huernia, whose carrion-scented flowers attract flies; Asian water vegetable Oenanthe javanica; and many plant genera,
families, and species, with California botanist Samuel Parish commemorated in plant names. Nepenthes comprises about 150 Old World tropical pitcher plants,
especially diverse in Borneo, Sumatra, and the Philippines; monkeys drink pitcher rainwater, and several Bornean species and hybrids are noted. Other highlights
include a
</item 8>

<item 9>
Consul fabius
Consul fabius, the tiger leafwing, is the most common and well known species of the genus "Consul" of subfamily Charaxinae in the brush-footed butterfly family
(Nymphalidae).

 It is found all over the Neotropics.
</item 9>

<item 10>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 10>
=== END ITEM id=883cf2f7bdd590e7 ===

=== ITEM id=dfccaa60838d278e ===
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
The Blizzard (1923 film)
The Blizzard is a 1923 Swedish drama film directed by Mauritz Stiller, starring Einar Hanson, Mary Johnson, Pauline Brunius and Hugo Björne.

 The film's original Swedish title is Gunnar Hedes saga, which means "The story of Gunnar Hede".

 The narrative revolves a student who tries to save his family's mansion which is facing bankruptcy.

 The film is loosely based on the Selma Lagerlöf novel "The Tale of a Manor".
</item 1>

<item 2>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 2>

<item 3>
Woman's film
The woman's film is a film genre which includes women-centered narratives, female protagonists and is designed to appeal to a female audience.

 Woman's films usually portray "women's concerns" such as problems revolving around domestic life, the family, motherhood, self-sacrifice, and romance.

 These films were produced from the silent era through the 1950s and early 1960s, but were most popular in the 1930s and 1940s, reaching their zenith during
World War II.

 Although Hollywood continued to make films characterized by some of the elements of the traditional woman's film in the second half of the 20th century, the
term itself disappeared in the 1960s.

 The work of directors George Cukor, Douglas Sirk, Max Ophüls, and Josef von Sternberg has been associated with the woman's film genre.

 Joan Crawford, Bette Davis, and Barbara Stanwyck were some of the genre's most prolific stars.
</item 3>

<item 4>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 4>

<item 5>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 5>

<item 6>
The Future of Oil: A Straight Story of the Canadian Oil Sands
The Future of Oil: A Straight Story of the Canadian Oil Sands
</item 6>

<item 7>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 7>

<item 8>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 8>

<item 9>
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
</item 9>

<item 10>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 10>
=== END ITEM id=dfccaa60838d278e ===

=== ITEM id=f599676c8902f53e ===
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
Graph query row:
person: Frederick King Keller
occupation: ['director', 'producer', 'screenwriter']
job_role: None
member_of: None
</item 1>

<item 2>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 2>

<item 3>
William Frederick King
William Frederick King CMG FRSC (February 19, 1854 – April 23, 1916) was a Canadian surveyor, astronomer, and civil servant.
</item 3>

<item 4>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 4>

<item 5>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 5>

<item 6>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 6>

<item 7>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 7>

<item 8>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 8>

<item 9>
Colby Keller
Colby Keller (born October 18, 1980) is an American visual artist, blogger, and pornographic film actor.

 His career in porn film started in 2004 at Sean Cody and has since expanded to include such studios as Cocksure Men, Randy Blue, Titan Men, Falcon, CockyBoys
and Men.com.

 Keller has amassed a large fan base with both his films and his long-standing blog, "The Big Shoe Diaries".

 He has been nominated for a number of awards and in 2013, World of Wonder awarded him the WOWie Award for Best Blog.

 Keller is nominated for 8 2015 Grabby Awards.

 Keller has also appeared in the short, Zolushka, a queer retelling of Cinderella and the popular series Capitol Hill both by the cult filmmaker Wes Hurley.
</item 9>

<item 10>
Graph query row:
person: Ian Schneider
occupation: ['independent film director', 'screenwriter']
job_role: None
member_of: None
</item 10>
=== END ITEM id=f599676c8902f53e ===
