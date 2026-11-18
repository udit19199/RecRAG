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

=== ITEM id=685a21e62290ffec ===
Gold sentences and facts:
Sentence 0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who successfully worked as the film director, the screenwriter, the art director and the animator.
  Fact 0.0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич) was born on 4 September 1924.
  Fact 0.1: Vladimir Petrovich Danilevich died on 9 October 2001.
  Fact 0.2: Vladimir Petrovich Danilevich was a well-known Soviet animator.
  Fact 0.3: Vladimir Petrovich Danilevich was a well-known Russian animator.
  Fact 0.4: Vladimir Petrovich Danilevich successfully worked as a film director.
  Fact 0.5: Vladimir Petrovich Danilevich successfully worked as a screenwriter.
  Fact 0.6: Vladimir Petrovich Danilevich successfully worked as an art director.
  Fact 0.7: Vladimir Petrovich Danilevich successfully worked as an animator.
Sentence 1: Wolfgang Becker (born 22 June 1954) is a German film director and writer.
  Fact 1.0: Wolfgang Becker was born on 22 June 1954.
  Fact 1.1: Wolfgang Becker is a German film director.
  Fact 1.2: Wolfgang Becker is a German writer.
Sentence 2:  He is best known to the international audience for his work "Good Bye Lenin!
  Fact 2.0: Wolfgang Becker is best known to the international audience for his work "Good Bye Lenin!".
Sentence 3: " (2003).
  Fact 3.0: "Good Bye Lenin!" is from 2003.

Retrieved items:
<item 1>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 1>

<item 2>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 2>

<item 3>
Farewell (1983 film)
Farewell (Russian: Прощание ) is a 1983 Soviet drama film based on Valentin Rasputin's novel "Farewell to Matyora" and directed by Elem Klimov.
</item 3>

<item 4>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 4>

<item 5>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 5>

<item 6>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 6>

<item 7>
Austrian director-screenwriter Stefan Ruzowitzky made *The Inheritors* (1998), Austria’s unsuccessful Oscar submission, and *The Counterfeiters* (2007), a WWII
drama about Nazi forgery at Sachsenhausen that won the foreign-language-film Oscar; Devid Striesow played Bernhard Krüger. German director Wolfgang Becker is
known for *Good Bye, Lenin!* (2003), starring Daniel Brühl and Katrin Saß. Other entries
</item 7>

<item 8>
Good Bye Lenin! (soundtrack)
Good Bye Lenin!

 is the original soundtrack album of the film of the same title starring Daniel Brühl and Katrin Sass.
</item 8>

<item 9>
Good Bye, Lenin!
Good Bye, Lenin!

 is a 2003 German tragicomedy film, directed by Wolfgang Becker.

 The cast includes Daniel Brühl, Katrin Saß, Chulpan Khamatova, and Maria Simon.

 Most scenes were shot at the Karl-Marx-Allee in Berlin and around Plattenbauten near Alexanderplatz.
</item 9>

<item 10>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 10>
=== END ITEM id=685a21e62290ffec ===

=== ITEM id=abe01acd877d275d ===
Gold sentences and facts:
Sentence 0: German Spitz is used to refer to both a breed of dog and category or type of dog.
  Fact 0.0: German Spitz is used to refer to a breed of dog.
  Fact 0.1: German Spitz is used to refer to a category or type of dog.
Sentence 1:  Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.
  Fact 1.0: Several modern breeds have been developed from the German Spitz.
  Fact 1.1: These modern breeds are either registered as separate breeds.
  Fact 1.2: These modern breeds are either registered as varieties of German Spitz.
Sentence 2:  The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that would later become known as the English Springer Spaniel.
  Fact 2.0: The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels.
  Fact 2.1: Attempts were made to use the term to specify a breed that would later become known as the English Springer Spaniel.

Retrieved items:
<item 1>
American Cocker Spaniel
The American Cocker Spaniel is a breed of sporting dog.

 It is a spaniel type dog that is closely related to the English Cocker Spaniel; the two breeds diverged during the 20th century due to differing breed
standards in America and the UK.

 In the United States, the breed is usually called the Cocker Spaniel, while elsewhere in the world, it is called the American Cocker Spaniel in order to
differentiate between it and its English cousin, which was already known as "Cocker Spaniel" before the American variety was created.

 The word "cocker" is commonly held to stem from their use to hunt woodcock in England, while "spaniel" is thought to be derived from the type's origins in
Spain.
</item 1>

<item 2>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
</item 2>

<item 3>
English Springer Spaniel
The English Springer Spaniel is a breed of gun dog in the Spaniel family traditionally used for flushing and retrieving game.

 It is an affectionate, excitable breed with a typical lifespan of twelve to fourteen years.

 They are very similar to the Welsh Springer Spaniel and are descended from the Norfolk or Shropshire Spaniels of the mid-19th century; the breed has diverged
into separate show and working lines.

 The breed suffers from average health complaints.

 The show-bred version of the breed has been linked to "rage syndrome", although the disorder is very rare.

 It is closely related to the Welsh Springer Spaniel and very closely to the English Cocker Spaniel; less than a century ago, springers and cockers would come
from the same litter.

 The smaller "cockers" hunted woodcock while the larger littermates were used to flush, or "spring," game.

 In 1902, The Kennel Club recognized the English Springer Spaniel as a distinct breed.

 They are used as sniffer dogs on a widespread basis.

 The term "Springer" comes from the historic hunting role, where the dog would flush (spring) birds into the air.
</item 3>

<item 4>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 4>

<item 5>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 5>

<item 6>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 6>

<item 7>
King Charles Spaniel
The King Charles Spaniel (also known as the English Toy Spaniel) is a small dog breed of the spaniel type.

 In 1903, the Kennel Club combined four separate toy spaniel breeds under this single title.

 The other varieties merged into this breed were the Blenheim, Ruby and Prince Charles Spaniels, each of which contributed one of the four colours available in
the breed.
</item 7>

<item 8>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 8>

<item 9>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 9>

<item 10>
Norfolk Spaniel
The Norfolk Spaniel or Shropshire Spaniel is an extinct breed of dog since the early 20th century.

 It was originally thought to have originated from the work of one of the Dukes of Norfolk, but this theory was disproven after being in doubt during the later
part of the 19th century.

 The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that
would later become known as the English Springer Spaniel.
</item 10>
=== END ITEM id=abe01acd877d275d ===

=== ITEM id=0e4d21df304df9ef ===
Gold sentences and facts:
Sentence 0:  He is the main vocalist of the boy group Highlight.
  Fact 0.0: The person described is the main vocalist of the boy group Highlight.
Sentence 1:  She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls' Generation, in 2007.
  Fact 1.0: The person described was a trainee at S.M. Entertainment's Starlight Academy during her middle school years.
  Fact 1.1: The person described debuted as a member of Girls' Generation, the girl group of S.M. Entertainment, in 2007.
  Fact 1.2: The person described debuted in Girls' Generation after her trainee years.

Retrieved items:
<item 1>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 1>

<item 2>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 2>

<item 3>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 3>

<item 4>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 4>

<item 5>
Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo"), also known as simply SJ or SUJU, is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 5>

<item 6>
List of songs written by Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo") is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 6>

<item 7>
Yang Yo-seob
Yang Yo-seob (born January 5, 1990), more commonly known as Yoseob, is a South Korean singer and musical actor.

 He is the main vocalist of the boy group Highlight.
</item 7>

<item 8>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 8>

<item 9>
Yang Yo-seob discography
The discography of South Korean singer-songwriter, musical actor Yang Yo-seob consists of one extended play and two singles.
</item 9>

<item 10>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 10>
=== END ITEM id=0e4d21df304df9ef ===

=== ITEM id=944e9e459a274f21 ===
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
Vector (band)
Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.

 The band had several drummers over the years, including Aaron Smith and Bruce Spencer, both of whom also played drums for The 77s.
</item 1>

<item 2>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 2>

<item 3>
Age of Chance
Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.

 They were perhaps most known for their mutant metallic cover of Prince's "Kiss" which topped the UK Indie Chart in 1986, and peaked at No. 50 in the UK Singles
Chart in January the following year.

 Despite signing for major label Virgin, and being favourites with the UK music press, they never enjoyed a major hit in the UK, although "Don't Get Mad… Get
Even" reached No. 8 on the US "Billboard" Hot Dance/Club Play chart.
</item 3>

<item 4>
Multivariate random variable
In probability, and statistics, a multivariate random variable or random vector is a list of mathematical variables each of whose value is unknown, either
because the value has not yet occurred or because there is imperfect knowledge of its value.

 The individual variables in a random vector are grouped together because they are all part of a single mathematical system — often they represent different
properties of an individual statistical unit.

 For example, while a given person has a specific age, height and weight, the representation of these features of "an unspecified person" from within a group
would be a random vector.

 Normally each element of a random vector is a real number.
</item 4>

<item 5>
Transstadial transmission
Transstadial transmission occurs when a pathogen remains with the vector from one life stage ("stadium") to the next.

 For example, the bacteria "Borrelia burgdorferi", the causative agent for Lyme disease, infects the tick vector as a larva, and the infection is maintained
when it molts to a nymph and later develops as an adult.

 This type of transmission is seen in other parasites like viruses or "Rickettsia".

 In addition to ticks, mites are another common vector.

 Some sources consider transstadial transmission a type of horizontal transmission, whereas other sources consider it vertical or partial vertical transmission.
</item 5>

<item 6>
Word2vec
Word2vec is a group of related models that are used to produce word embeddings.

 These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.

 Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus
being assigned a corresponding vector in the space.

 Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the
space.
</item 6>

<item 7>
Christopher Evans (author)
Christopher Evans (born 1951 in Tredegar, Wales) is a British science fiction writer and children's author.

 His novels include "Capella's Golden Eyes" (1980), "The Insider" (1981), "Mortal Remains" (1995) and "Ice Tower" (2000).

 He is the co-editor (with Robert Holdstock) of three original SF anthologies, "Other Edens" (1987), "Other Edens II" (1988) and "Other Edens III" (1989).
</item 7>

<item 8>
Kinetic Records
Kinetic Records was a New York City-based record label founded by Steve Lau (founding member of Sire Records act the Ocean Blue), that launched the careers of
artists and DJs such as Paul Oakenfold, BT, Ray Munns, Deepsky, Kosheen, Tilt, Timo Maas, Grace, Sandra Collins, Billy Thermal, Binary Finary, and Shpongle.

 Other successful artists on the label included Ash, Sasha and John Digweed, South, Faithless, DJ DAN, LTJ Bukem, Dario G, Luomo, Infected Mushroom, Max Graham,
Dave Ralph Music Instructor, and Hybrid.

 It was also well known for its Tranceport series, of which Paul Oakenfold's first installment is considered to be the breakthrough record for the trance genre
with the highest sales of any mix compilation.

 The release of Sasha's Airdrawndagger and Sasha and Digweed's mix compilation "Communicate" were also big releases for the label.

 Kinetic was the exclusive distributor of the Perfecto label in the US as well as the Another Late Night compilation series, which featured Howie B., Fila
Brazillia, and Rae and Christian.
</item 8>

<item 9>
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
</item 9>

<item 10>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 10>
=== END ITEM id=944e9e459a274f21 ===

=== ITEM id=5f57af42286f85a1 ===
Gold sentences and facts:
Sentence 0: Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.
  Fact 0.0: The Dogo Cubano is also known as the Cuban Mastiff.
  Fact 0.1: The Dogo Cubano is also known as the Cuban Dogo.
  Fact 0.2: The Dogo Cubano is also known as the Cuban Dogge.
  Fact 0.3: The Dogo Cubano is an extinct dog breed.
  Fact 0.4: The Dogo Cubano is a dog breed from Cuba.
Sentence 1: The Dutch Shepherd is a herding dog of Dutch origin.
  Fact 1.0: The Dutch Shepherd is a herding dog.
  Fact 1.1: The Dutch Shepherd is a herding dog of Dutch origin.

Retrieved items:
<item 1>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 1>

<item 2>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 2>

<item 3>
Graph query row:
entity: Dutch Shepherd
categories: ['__KGBuilder__', '__Entity__', 'Thing']
kind: breed
</item 3>

<item 4>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 4>

<item 5>
Dogo Cubano
Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.

 It was of the Bull Mastiff type and was used for dog fighting.
</item 5>

<item 6>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 6>

<item 7>
Bulldog
The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.

 Other scent-hound breeds include the Small Greek Domestic Dog, Irish Wolfhound, Bluetick Coonhound, Finnish Lapphund, and the Basset Hound.

 The Bulldog is a muscular, hefty dog with a wrinkled face and a distinctive pushed-in nose.

 The American Kennel Club (AKC), The Kennel Club (UK), and the United Kennel Club (UKC) oversee breeding records.

 Bulldogs were the fourth most popular purebreed in the US in 2007 according to the American Kennel Club.
</item 7>

<item 8>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 8>

<item 9>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 9>

<item 10>
Romanian Raven Shepherd Dog
The Romanian Raven Shepherd Dog is a very large Romanian livestock guardian dog, taxonomized within the second group of dog breeds - Pinscher and Schanuzer -
Molossoid Breeds - Swiss Mountain and Cattle Dogs, section 2.2: Molossoid Breeds - Mountain type.
</item 10>
=== END ITEM id=5f57af42286f85a1 ===

=== ITEM id=db44c65999a57e39 ===
Gold sentences and facts:
Sentence 0: Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and medical teaching facilities in Saint John, New Brunswick.
  Fact 0.0: Dalhousie University, commonly known as Dal, is a public research university.
  Fact 0.1: Dalhousie University is in Nova Scotia, Canada.
  Fact 0.2: Dalhousie University has three campuses in Halifax.
  Fact 0.3: Dalhousie University has a fourth campus in Bible Hill.
  Fact 0.4: Dalhousie University has medical teaching facilities in Saint John, New Brunswick.
Sentence 1: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 1.0: California State Polytechnic University, Pomona is also known as CPP, Cal Poly Pomona, or Cal Poly.
  Fact 1.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 1.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.

Retrieved items:
<item 1>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 1>

<item 2>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 2>

<item 3>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 3>

<item 4>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 4>

<item 5>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 5>

<item 6>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 6>

<item 7>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 7>

<item 8>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 8>

<item 9>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=db44c65999a57e39 ===

=== ITEM id=0e1b10c49d256c17 ===
Gold sentences and facts:
Sentence 0: Essence is a monthly magazine for African American women between the ages of 18 and 49.
  Fact 0.0: Essence is a monthly magazine.
  Fact 0.1: Essence is for African American women.
  Fact 0.2: Essence is aimed at women between the ages of 18 and 49.
Sentence 1: ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
  Fact 1.0: ALT for Damerne is a women's magazine.
  Fact 1.1: ALT for Damerne means 'All for the Ladies' in English.
  Fact 1.2: ALT for Damerne is written in the Danish language.
  Fact 1.3: ALT for Damerne is a weekly magazine.
  Fact 1.4: ALT for Damerne is published in Copenhagen, Denmark.

Retrieved items:
<item 1>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 1>

<item 2>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 2>

<item 3>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 3>

<item 4>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 4>

<item 5>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 5>

<item 6>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 6>

<item 7>
Essence Music Festival
The Essence Festival, known as "the party with a purpose", is an annual music festival which started in 1995 as a one-time event to celebrate the 25th
anniversary of "Essence", a magazine aimed primarily towards African-American women.

 It is the largest event celebrating African-American culture and music in the United States.

 Locally referred to as the Essence Fest, it has been held in New Orleans, Louisiana every year since 1994 except for 2006, when it was held in Houston, Texas
due to Hurricane Katrina's effect on New Orleans.

 It also held in Durban, KwaZulu-Natal in 2016.

 It features artists simultaneously performing on a main stage as well as four standing-room only superlounge stages.
</item 7>

<item 8>
Alt for Damerne
ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
</item 8>

<item 9>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 9>

<item 10>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 10>
=== END ITEM id=0e1b10c49d256c17 ===

=== ITEM id=036ee2c5d18bca26 ===
Gold sentences and facts:
Sentence 0: Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public school students under the First Amendment to the United States Constitution.
  Fact 0.0: Minersville School District v. Gobitis, 310 U.S. 586 (1940), was a decision by the Supreme Court of the United States.
  Fact 0.1: Minersville School District v. Gobitis involved the religious rights of public school students.
  Fact 0.2: Minersville School District v. Gobitis involved religious rights under the First Amendment to the United States Constitution.
Sentence 1: Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.
  Fact 1.0: Gravel v. United States, 408 U.S. 606 (1972), was a case.
  Fact 1.1: Gravel v. United States concerned the protections offered by the Speech or Debate Clause of the United States Constitution.

Retrieved items:
<item 1>
Forest Grove School District v. T. A.
Forest Grove School District v. T. A., 557 U.S. 230 (2009), is a case in which the United States Supreme Court held that the Individuals with Disabilities
Education Act (IDEA) authorizes reimbursement for private special education services when a public school fails to provide a "free appropriate public education"
(FAPE) and the private school placement is appropriate, regardless of whether the child previously received special education services through the public
school.
</item 1>

<item 2>
Cedar Rapids Community School Dist. v. Garret F.
Cedar Rapids Community School District v. Garret F., 526 U.S. 66 (1999) , was a United States Supreme Court case in which the Court ruled that the related
services provision in the Individuals with Disabilities Education Act (IDEA) required public school districts to fund "continuous, one-on-one nursing care for
disabled children" despite arguments from the school district concerning the costs of the services."

 The judges relied heavily on Irving Independent School Dist.

 v. Tatro, 468 U. S. 883 (1984).

 Under the Court's reading of the IDEA's relevant provisions, medical treatments such as suctioning, ventilator checks, catheterization, and others which can be
administered by non-physician personnel come within the parameters of the special education law's related services.

 Disability advocates considered the Court decision to be a "substantial victory for families of children with disabilities."

 Amendments were made in the Education Flexibility Partnership Act of 1999 to increase IDEA funding as a result of the case.
</item 2>

<item 3>
Safford Unified School District v. Redding
Safford Unified School District v. Redding, 557 U.S. 364 (2009) , was a United States Supreme Court case in which the Court held that a strip search of a middle
schooler violated the Fourth Amendment to the United States Constitution where the school lacked reasons to suspect either that the drugs (Ibuprofen) presented
a danger or that they were concealed in her underwear.

 The court also held, however, that because this was not clearly established law prior to the court's decision, the officials involved were shielded from
liability by qualified immunity.
</item 3>

<item 4>
Bethel School District v. Fraser
Bethel School District v. Fraser, 478 U.S. 675 (1986), was a landmark decision by the United States Supreme Court involving free speech in public schools.

 High school student Matthew Fraser was suspended from school in the Bethel School District for making a speech including sexual double entendres at a school
assembly.

 The Supreme Court held that his suspension did not violate the First Amendment.
</item 4>

<item 5>
Elk Grove Unified School District v. Newdow
Elk Grove Unified School District v. Newdow, 542 U.S. 1 (2004), was a case decided by the U.S. Supreme Court.

 The lawsuit, originally filed as Newdow v. United States Congress, Elk Grove Unified School District, et al. in 2000, led to a 2002 ruling by the United States
Court of Appeals for the Ninth Circuit that the words "under God" in the Pledge of Allegiance are an endorsement of religion and therefore violate the
Establishment Clause of the First Amendment to the United States Constitution.

 After an initial decision striking the congressionally added "one nation under God" language, [Newdow v. U.S. Congress, 292 F.3d 597 (9th Cir.

 2002), the superseding opinion on denial of rehearing en banc was more limited, holding that compelled recitation of the language by school teachers to
students was invalid.

 [328 F.3d 466 (9th Cir.

 2003).
</item 5>

<item 6>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 6>

<item 7>
Gravel v. United States
Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.

 In the case, the Supreme Court of the United States held that the privileges and immunities of the Constitution's Speech or Debate Clause enjoyed by members of
Congress also extend to Congressional aides, but not to activity outside the legislative process.
</item 7>

<item 8>
Minersville School District v. Gobitis
Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public
school students under the First Amendment to the United States Constitution.

 The Court ruled that public schools could compel students—in this case, Jehovah's Witnesses—to salute the American Flag and recite the Pledge of Allegiance
despite the students' religious objections to these practices.

 This decision led to increased persecution of Witnesses in the United States.

 The Supreme Court overruled this decision a mere three years later, in "West Virginia State Board of Education v. Barnette", 319 U.S. 624 (1943).
</item 8>

<item 9>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 9>

<item 10>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 10>
=== END ITEM id=036ee2c5d18bca26 ===

=== ITEM id=d7fa8114fa78044e ===
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
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 1>

<item 2>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 2>

<item 3>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 3>

<item 4>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 4>

<item 5>
Augusta National Golf Club
Augusta National Golf Club, located in Augusta, Georgia, is one of the most famous golf clubs in the world.

 Founded by Bobby Jones and Clifford Roberts on the site of the former Fruitland (later Fruitlands) Nursery, the course was designed by Jones and Alister
MacKenzie and opened for play in January 1933.

 Since 1934, it has played host to the annual Masters Tournament, one of the four major championships in professional golf, and the only major played each year
at the same course.

 It was the number one ranked course in "Golf Digest"<nowiki>'</nowiki>s 2009 list of America's 100 greatest courses and is currently the number ten ranked
course on "Golfweek Magazine"' s 2011 list of best classic courses in the United States, in terms of course architecture.
</item 5>

<item 6>
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
</item 6>

<item 7>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 7>

<item 8>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 8>

<item 9>
Sheshan Golf Club
Sheshan Golf Club () is the first premier private golf club in Shanghai, China.

 Founded in 2004, the 18 hole golf course plays to a par of 72 and is designed by Nelson and Haworth Design, and spans over 7,266 yards (6,531 meters).

 Every year, the club plays host to the WGC-HSBC Champions, and is currently the only club to host the Championship.

 Sheshan Golf Club was awarded the Best Golf Course in China by "Golf Digest" in 2013 and currently tops of the list of Top Clubs in China by "Golf Magazine".
</item 9>

<item 10>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 10>
=== END ITEM id=d7fa8114fa78044e ===

=== ITEM id=58b95747ad0bc49b ===
Gold sentences and facts:
Sentence 0: Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.
  Fact 0.0: Midland F1 Racing was a Formula One constructor.
  Fact 0.1: Midland F1 Racing was a racing team.
  Fact 0.2: Midland F1 Racing is also known as MF1 Racing.
Sentence 1:  In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
  Fact 1.0: In 2007, the team competed as Spyker F1.
  Fact 1.1: In 2008, the team was sold to Indian businessman Vijay Mallya.
  Fact 1.2: In 2008, the team was renamed Force India F1.
Sentence 2: The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.
  Fact 2.0: The Spyker F1 Team was known as the Etihad Aldar Spyker F1 Team for sponsorship reasons.
  Fact 2.1: The Spyker F1 Team was a Formula One team.
  Fact 2.2: The Spyker F1 Team competed in the 2007 Formula One World Championship.
  Fact 2.3: The Spyker F1 Team was created by Spyker Cars after their buyout of the short-lived Midland F1 team.
  Fact 2.4: Midland F1 was formerly known as Jordan Grand Prix.

Retrieved items:
<item 1>
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 1>

<item 2>
Marussia F1
The Marussia F1 Team (subsequently Manor Marussia F1 Team) was an Anglo-Russian Formula One racing team and constructor which was based in Banbury, Oxfordshire
in the United Kingdom.

 The team was operated by Manor Motorsport (formerly Marussia Manor Racing), which was previously a subsidiary of Marussia Motors, a now defunct sports car
manufacturer which was based in Moscow.

 The team originally started racing in under the "Virgin Racing" name; the following year Virgin adopted Marussia as a title sponsor becoming "Marussia Virgin
Racing" until being fully rebranded as the "Marussia F1 Team" for .
</item 2>

<item 3>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 3>

<item 4>
Simtek
Simtek (Simulation Technology) was an engineering consultancy firm and Formula One racing team.

 The Formula One (F1) engineering consultancy arm, Simtek Research, was founded in 1989 by Max Mosley and Nick Wirth.

 It originally was involved in many areas of Formula One, including wind tunnel construction and chassis building for third parties.

 Simtek Grand Prix, the racing team, launched in 1993 and competed in the 1994 and 1995 seasons achieving a best result of ninth place.

 With large debts and a lack of sponsorship money, Simtek went into voluntary liquidation in June 1995.
</item 4>

<item 5>
Midland F1 Racing
Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.

 It competed in the 2006 Formula One season with drivers Christijan Albers and Tiago Monteiro.

 The team was created by the renaming of Jordan Grand Prix after its purchase by Canadian businessman, and owner of the Midland Group, Alex Shnaider.

 The team was registered as the first Russian Formula One team, reflecting Shnaider's roots, although it continued to be based in the United Kingdom, at
Jordan's Silverstone factory.

 Towards the end of the 2006 season, the team was sold to Spyker Cars N.V.; the team raced in its last three Grands Prix under the official name "Spyker MF1
Racing".

 In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
</item 5>

<item 6>
Jaguar R3
The Jaguar R3 is a Formula One racing car with which Jaguar Racing competed in the 2002 Formula One season.

 The car was launched on January 4, 2002.

 It was driven by Eddie Irvine and Pedro de la Rosa, both retained from 2001.
</item 6>

<item 7>
Jordan 195
The Jordan 195 was a Formula One car for the 1995 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 The team never employed a test driver.

 The engine was a Peugeot A10 3.0 V10.

 The team's main sponsor was Total.
</item 7>

<item 8>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 8>

<item 9>
Spyker F8-VII
The Spyker F8-VII (subsequently known as the Force India VJM01) was a Formula One car, constructed by Spyker F1 that competed in the 2007 Formula One World
Championship.

 A "B Specification" car named the Spyker F8-VIIB was launched at the Italian Grand Prix and used for the remainder of the 2007 season.

 The engine of Spyker F8-VII car was Ferrari 056 despite the team opted for 2006-spec engine due to cost reasons.

 For the 2008 World Championship, Force India used a slightly modified version of the F8-VIIB, called the Force India VJM01 named after team owners Vijay
Mallya, Jan Mol and Michiel Mol.

 The VJM01 used 2007-spec Ferrari 056 engines instead of 2008-spec.

 The F8-VII was the only car constructed by Spyker F1 in their own right after their take-over from the struggling Midland F1 team part way through 2006.
</item 9>

<item 10>
F1 2000 (video game)
F1 2000 is a racing video game based on the 2000 Formula One season, developed by EA Sports and released for the PlayStation and Microsoft Windows formats.

 "F1 2000" was the last Visual Sciences F1 racing game to appear on the "PlayStation".

 With an official FIA Formula One license, it includes the full 2000 world championship season, including the new Indianapolis circuit and the Jaguar Racing
team.

 A rival game, "F1 Championship Season 2000", was released on December 23, 2000 for PC, PlayStation, PlayStation 2, Macintosh and Game Boy Color.
</item 10>
=== END ITEM id=58b95747ad0bc49b ===

=== ITEM id=b3246b80bb7a4cc1 ===
Gold sentences and facts:
Sentence 0: Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a 1760 play by Francesco Ringhieri, "L'Osiride".
  Fact 0.0: Mosè in Egitto (Moses in Egypt) is a three-act opera.
  Fact 0.1: Mosè in Egitto was written by Gioachino Rossini.
  Fact 0.2: Mosè in Egitto has an Italian libretto by Andrea Leone Tottola.
  Fact 0.3: The libretto of Mosè in Egitto was based on a 1760 play by Francesco Ringhieri.
  Fact 0.4: The 1760 play by Francesco Ringhieri is titled L'Osiride.
Sentence 1: Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.
  Fact 1.0: Lucrezia Borgia is a melodramatic opera.
  Fact 1.1: Lucrezia Borgia is in a prologue and two acts.
  Fact 1.2: Lucrezia Borgia is by Gaetano Donizetti.

Retrieved items:
<item 1>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 1>

<item 2>
Lucrezia Borgia (1912 film)
Lucrezia Borgia is a 1912 Italian silent historical film directed by Gerolamo Lo Savio and starring Francesca Bertini in the title role of Lucrezia Borgia.
</item 2>

<item 3>
Lucrezia Borgia (1940 film)
Lucrezia Borgia is a 1940 Italian historical film directed by Hans Hinrich and starring Isa Pola, Friedrich Benfer and Carlo Ninchi.

 The film portrays the life of Lucrezia Borgia (1480-1519), one of a number of Italian films of the era set during the Renaissance.

 It was made at the Scalera Studios in Rome.
</item 3>

<item 4>
Lucrezia Borgia (1922 film)
Lucrezia Borgia is a 1922 German silent historical film directed by Richard Oswald and starring Conrad Veidt, Liane Haid and Albert Bassermann.

 It was based on a novel by Harry Sheff, and portrayed the life of the Renaissance Italian aristocrat Lucrezia Borgia (1480-1519).

 Botho Hoefer and Robert Neppach worked as the film's art directors, designing the period sets needed.
</item 4>

<item 5>
Mosè in Egitto
Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a
1760 play by Francesco Ringhieri, "L'Osiride".

 It premièred on 5 March 1818 at the recently reconstructed Teatro San Carlo in Naples, Italy.
</item 5>

<item 6>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 6>

<item 7>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
</item 7>

<item 8>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 8>

<item 9>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 9>

<item 10>
Lucrezia Borgia (1947 film)
Lucrezia Borgia (Spanish:Lucrecia Borgia) is a 1947 Argentine historical film directed by Luis Bayón Herrera and starring Olinda Bozán, Héctor Quintanilla and
Gogó Andreu.

 The film portrays the life of Lucrezia Borgia (1480-1519).
</item 10>
=== END ITEM id=b3246b80bb7a4cc1 ===

=== ITEM id=73ebee6ef358116f ===
Gold sentences and facts:
Sentence 0: Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.
  Fact 0.0: Wenling is a coastal county-level city.
  Fact 0.1: Wenling is in the municipal region of Taizhou.
  Fact 0.2: Taizhou is in southeastern Zhejiang province.
  Fact 0.3: Wenling is in China.
  Fact 0.4: In the Wenling dialect, Wenling is called Uen-lin Zy.
Sentence 1: Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.
  Fact 1.0: Ruili is a county-level city of Dehong Prefecture.
  Fact 1.1: Ruili is in the west of Yunnan province.
  Fact 1.2: Yunnan province is in the People's Republic of China.
  Fact 1.3: Ruili's Burmese name is ရွှေလီ.
  Fact 1.4: Ruili's Thai name is เมืองมาว.

Retrieved items:
<item 1>
Dali–Ruili Railway
The Dali–Ruili Railway or Darui Railway (), is a single-track electrified railroad under construction in Yunnan Province of Southwest China.

 The line is slated to run 336.39 km from Dali in central Yunnan to Ruili in southwestern Yunnan on the border with Myanmar.

 The line traverses rugged terrain, and bridges and tunnels will account for 75% of the total track length, including a 36-km tunnel through the Gaoligong
Mountains.

 Construction began in May 2011 and is scheduled to take six years.

 Cities and towns along route include Dali, Yangbi Yi Autonomous County, Yongping County, Baoshan, Mang City (also known as Mangshi or Dehong) and Ruili.
</item 1>

<item 2>
Wanding Border Economic Cooperation Zone
Wanding Border Economic Cooperation Zone (WTBECZ) is a Chinese State Council-approved Industrial Park based in Wanding Town, Ruili City, Dehong Prefecture,
Yunnan, China, founded in 1992 and was established to promote trade between China and Myanmar.

 The zone spans 6 km sq. and is focuses on developing trading, processing, agriculture resources and tourism.
</item 2>

<item 3>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 3>

<item 4>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Ruili Border Economic Cooperation Zone
Ruili Border Economic Cooperation Zone (RLBECZ) is a Chinese State Council-approved Industrial Park based in Ruili City, Dehong Prefecture, Yunnan, China,
founded in 1992 and was established to promote trade between China and Myanmar.

 The zone connects Ruili with Man-wing City.
</item 6>

<item 7>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 7>

<item 8>
G56 Hangzhou–Ruili Expressway
The Hangzhou–Ruili Expressway (), commonly referred to as the "Hangrui Expressway" () is an expressway in China that connects the cities of Hangzhou, Zhejiang,
and Ruili, Yunnan, a city on the border with Burma.

 When complete, it will be 2935 km in length.
</item 8>

<item 9>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 9>

<item 10>
Ruili
Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.

 It is a major border crossing between China and Myanmar, with the town of Muse located across the border.
</item 10>
=== END ITEM id=73ebee6ef358116f ===
