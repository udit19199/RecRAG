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

This file has 7 items.

=== ITEM id=096577dd00eb9cad ===
Gold sentences and facts:
Sentence 0: No . Official English title Original Japanese title Original air date English air date 151 `` Sabertooth '' `` Seibātūsu '' ( 剣 咬 の 虎 ( セイバー トゥース ) ) October 6 , 2012 November 17 , 2015 The Legion Platoon is tasked by Zentopia 's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again . Before embarking on their mission , they visit Fairy Tail and make amends with the guild for their previous hostility against them . It is later discovered that Fairy Tail has lost their title as the strongest guild in Fiore to the Saber Tooth guild over the past seven years . Two of the strongest wizards in the guild include a pair of Dragon Slayers named Sting and Rogue , who are obsessed with fighting Natsu and Gajeel , respectively . Meanwhile , Makarov brings Gildarts to their abandoned guildhall and shows him Lumen Histoire , a light that Makarov says is Fairy Tail 's greatest secret . Makarov then tells Gildarts that he has been chosen to become the next master of Fairy Tail . 152 `` And So We Aim for the Top '' `` Soshite Ore - tachi wa Chōjō o Mezasu '' ( そして オレ たち は 顶 上 を 目指す ) October 13 , 2012 November 24 , 2015 Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island . The group meets with Poluchka for medicine to strengthen their magic , but are unsuccessful . Wendy recognizes Poluchka 's voice and scent as those of Grandine , the dragon who raised her . Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her . Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 `` Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction . When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years . They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild , Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power . The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave . 155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray , Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus . As they tour the city , Natsu and Happy encounter the twin dragon slayers , Sting and Rogue , who mock Natsu for not being able to kill Acnologia , and boast of becoming true dragon slayers by killing the dragons who raised them . At midnight , Fairy Tail and the other participating guilds are put through a preliminary event -- a race through a massive floating labyrinth -- designed to narrow the number of participating guilds down to eight teams . 156 `` Sky Labyrinth '' `` Sukai Rabirinsu '' ( 空中 迷宮 ( スカイ ラビリンス ) ) November 10 , 2012 December 22 , 2015 Natsu 's team notices that Wendy has gone missing , so Elfman replaces her to keep them from being eliminated . As they make their way through the enchanted labyrinth , the five encounter various other guilds and steal the maps made by the other guilds to better navigate the maze . They manage to reach the end of the maze and pass the elimination round , confident they have arrived first , only to discover they have barely taken last place . Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace , Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla . To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced , they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars '' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once . Humiliated , Gray swears revenge against Raven Tail and Saber Tooth . Afterwards , Lucy is chosen to fight Raven Tail member Flare in the tournament 's first battle round . 159 `` Lucy vs. Flare '' `` Rūshii vs. Furea '' ( ルーシィ vs. フレア ) December 1 , 2012 January 12 , 2016 Lucy displays tremendous skill in her battle against Flare by being able to summon two spirits simultaneously . As Lucy gains the upper hand , Flare cheats by using her magic hair to hold Alzack and Bisca 's daughter Asuka as a hostage , blackmailing Lucy into not fighting back to keep her from harming Asuka . As Lucy tearfully speaks of Asuka 's danger , Natsu is able to hear it and rushes to Asuka to incinerate Flare 's hair , allowing Lucy to counterattack with her Urano Metoria spell in combination with Gemini . However , before the spell fully activates , Flare 's teammate Obra nullifies Lucy 's magic , causing her to collapse and lose the match . 160 `` Portent '' `` Kyōzui '' ( 凶 瑞 ) December 8 , 2012 January 19 , 2016 Blue Pegasus ' Ren defeats Mermaid Heel 's Araña during the games ' second match , while Saber Tooth 's Orga effortlessly defeats Quatro Cerberus ' War Cry in the third match . For the day 's fourth and final match , Jellal is pitted against Lamia Scale 's Jura , intending to avenge the humiliated Fairy Tail members . He mimics Mystogan 's magic to keep up his disguise , but he finds it ineffective to use against Jura and is forced to use his own magic instead . Before he can cast a spell that will expose his true identity , Ultear uses Merudy 's sensory to force Jellal into submission . Meanwhile , the hospitalized Carla awakens after having a troubling premonition of the future . 161 `` Chariots '' `` Chariotto '' ( 战 车 ( チャリオット ) ) December 15 , 2012 January 26 , 2016 Fairy Tail is undaunted by their poor performance on the first day . As the guild celebrates , they are visited and taunted by Bacchus , Erza 's rival from Quatro Cerberus , who has come to replace War Cry . Carla relays her premonition to Poluchka , saying that she saw Mercurius collapsing around what appeared to be a tearfully singing Lucy . The second day of the tournament begins with a footrace atop a train of moving chariots , causing the participants Natsu , Gajeel , and Sting to suffer from motion sickness as a side effect of their Dragon Slayer magic . As Natsu and Gajeel force themselves forward , Sting inquires about their determination to which Natsu responds that it is as a result of the strength of Fairy Tail 's bonds . Sting eventually throws the race while Natsu and Gajeel manage to finish as sixth and seventh placers , respectively , earning Fairy Tail the audience 's respect . 162 `` Elfman vs. Bacchus '' `` Erufuman vs. Bakkasu '' ( エルフマン vs. バッカス ) December 22 , 2012 September 1 , 2015 Due to a miscomunnication between the king and his knight Arcadios , Elfman is chosen as Bacchus ' opponent instead of Erza . Before the match commences , the two opponents make a wager : if Bacchus wins , he will court Elfman 's sisters , and if Elfman wins , Bacchus ' team name will be changed to `` Quatro Puppy '' for the rest of the games . Bacchus ruthlessly attacks Elfman with his bare palms , who manages to damage Bacchus with his defensive Lizardman transformation . Elfman manages to outlast Bacchus as he exhausts himself , winning the match and forcing Bacchus ' team to be renamed . Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his `` Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts . Jenny then transforms into her battle armor while Mirajane shifts into her Satan Soul `` Sitri '' form and easily defeats Jenny . The final match of the day is announced , with Kagura Mikazuchi of Mermaid Heel pitted against Yukino Aguria of Saber Tooth . Yukino proposes to Kagura that they wager their lives on their battle . Meanwhile , Arcadios is confronted by Darton , the kingdom 's defense minister , about his `` Eclipse Project '' . 164 `` Kagura vs. Yukino '' `` Kagura vs. Yukino '' ( カグラ vs. ユキノ ) January 12 , 2013 September 1 , 2015 Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura . Using her gravity - altering magic , Kagura defeats the two spirits , thus forcing Yukino to use a Gatekey that summons Ophiuchus , which is regarded as the strongest among the Celestial Spirits . However , Kagura slays Ophiuchus with her sword still sheathed and easily defeats Yukino . Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance . Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete '' ( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ; however , Lucy turns down the offer . Meanwhile , Milliana tells Erza that Kagura 's sword , Archenemy , is intended to kill Jellal , and that she and Kagura share the same resentment towards Jellal . Natsu and Happy run after Yukino in an attempt to apologize ; however , Natsu realizes that Yukino had been expelled , and infiltrates the Crocus Garden to fight Jiemma . 166 `` Pandemonium '' `` Pandemoniumu '' ( 伏魔殿 ( パンデモニウム ) ) January 26 , 2013 September 1 , 2015 After Natsu punches out Dobengal , the guild member summoned to fight him , he and Jiemma engage in a duel . However , the guild master 's daughter Minerva appears , ends their duel and , taking Happy hostage , convinces Natsu to settle their dispute in the Grand Magic Games . As Natsu leaves , he states that Saber Tooth will never surpass Fairy Tail because they care for their friends , words that are echoed by Rogue , while Sting is left in awe at Natsu 's strength . The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points . Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent scores depending on the extent of power used . In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei '' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve . For the third match , Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail . Since the opponent is from a suspicious guild , the Fairy Tail members take extra precautions by watching the actions of Ivan Dreyar as well as the other four Raven Tail members . However , as the match commences , the spectators are shocked to see Laxus being beaten by Alexei , although no cautious movements can be seen from the Raven Tail team . It turns out that the two dueling people are just illusions created by Alexei in order for him to have a talk with Laxus . Also , the other Raven Tail members being monitored by Fairy Tail are just lifelike illusions , for in reality they are with Alexei in the battlefield . Alexei reveals himself to be Ivan Dreyar . Unable to interrogate Laxus of the Lumen Histoire 's location , Ivan and the Raven Tail members are forced to fight him . Before their battle begins , Ivan declares that Raven Tail is an anti-Fairy Tail guild . 169 `` Wendy vs. Shelia '' `` Wendi vs. Sheria '' ( ウェンディ vs. シェリア ) February 16 , 2013 September 1 , 2015 After Laxus defeats Ivan and the four Raven Tail elite members , Ivan 's illusions disappear to the shock of everyone in the stadium . The game officials declare Laxus the victor , and Ivan tells him that Lumen Histoire is Fairy Tail 's ' darkness ' . Due to the infraction of the rules , the Raven Tail team is disqualified for the remainder of the Games , and its members are captured by the soldiers . Later , Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off to compete for the final match of the day . As the two exchange attacks , gusts of wind surround the arena since Sherria is revealed to be a Sky God Slayer . Wendy employs an advanced Sky Dragon Slayer attack much to Poluchka 's astonishment , causing Sherria to fall on the ground . Wendy is about to be announced as the winner of the bout when Sherria suddenly stands up with her wounds fully healed . Meanwhile , the Crime Sorcière guild senses the nefarious aura , and Jellal proceeds to the Domus Flau audience , oblivious that he bumped into Doranbolt , who , in turn , wonders what is Mystogan doing in Earthland . 170 `` Small Fists '' `` Chiisana Kobushi '' ( 小さな 拳 ) February 23 , 2013 September 1 , 2015 With Cheria being fully healed and Wendy being almost depleted of magic power , their battle continues , with the former casting a high - level spell towards the latter . However , it seems that Wendy enhances Cheria 's physical strength , causing the attack to gain much force and thus missing Wendy . The two competitors then engage in a hand - to - hand combat , but due to the time limit and neither side is giving up , their battle resulted in a draw , giving both teams with 5 points . Meanwhile , Jellal manages to track the sinister aura among the crowd in the form of a mysterious , hooded person . Jellal is about to confront the aura source , but he is abruptly cornered by Doranbolt , Lahar , and other Rune Knights . Doranbalt then interrogates Jellal , and as his cover was blown , the Rune Knights , as well as Kagura and Milliana glancing nearby , are startled to see the true identity of Mystogan . Yajima convinces the Rune Knights that Mystogan is Jellal 's Edolas counterpart , prompting Lahar to release the fugitive . Meanwhile , Laxus and Makarov have a small conversation regarding Lumen Histoire when Mavis appears and explains that Lumen Histoire is the ' light ' of Fairy Tail . 171 `` Naval Battle '' `` Nabaru Batoru '' ( 海战 ( ナバルバトル ) ) March 2 , 2013 September 1 , 2015 As the jubilant Fairy Tail members celebrate their victories , the other people , such as Sting , Kagura , and Milliana , are contemplating on their own concerns , while Yukino is enlisted as a sergeant under Arcadios ' command . Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others rush to her aid . 172 `` A Parfum for You '' `` Kimi ni Sasageru Parufamu '' ( 君 に 捧げる 香り ( パルファム ) ) March 9 , 2013 September 1 , 2015 While Lucy is hospitalized and taken to the infirmary , the Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so . Later , Makarov announces that the two Fairy Tail teams must merge as a result of the disqualification of Raven Tail and to compensate for the odd number of teams for the battle round . He also reveals that the new team will have the former Team A 's 35 points to start with . The new Fairy Tail team , consisting of Natsu , Gray , Erza , Gajeel , and Laxus , are well - received as they enter the field . The second part of the tournament begins with a tag battle between Blue Pegasus ' Ichiya and `` Rabbit '' and Quatro Puppy 's Rocker and Bacchus . To everyone 's shock , it is revealed that `` Rabbit '' is actually Nichiya in disguise , and after the Exceed is punched by Bacchus , Ichiya bulks up to defeat the Quatro Puppy opponents , awarding Blue Pegasus with 10 points . 173 `` Battle of Dragon Slayers ! '' `` Batoru obu Doragon Sureiyā '' ( バトル ・ オブ ・ ドラゴン スレイヤー ) March 16 , 2013 September 1 , 2015 Mermaid Heel 's Kagura and Milliana and Lamia Scale 's Lyon and Yuka face off for the second tag battle of the games . The two teams struggle to earn a victory , only to have their battle ending in a draw due to the time limit . For the third and final match of the day , the spectators look on in silence as Natsu and Gajeel of Team Fairy Tail and Sting and Rogue of Team Saber Tooth enter the field . As the battle commences , Sting and Rogue were about to take the initiative when they were suddenly attacked by Natsu and Gajeel . The four mages then exchange attacks ; however , the audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers . Unable to keep themselves on par with their adversaries , Sting activates ' White Drive ' while Rogue powers up using ' Shadow Drive ' . 174 `` Four Dragons '' `` Yonin no Doragon '' ( 四 人 の 竜 ( ドラゴン ) ) March 23 , 2013 September 1 , 2015 The tag battle continues , with Sting and Rogue beginning to attack Natsu and Gajeel with their powers being increased . After receiving heavy blows from the Saber Tooth Dragon Slayers , Natsu and Gajeel manages to hold them off , thus forcing Sting and Rogue to activate `` Dragon Force '' on their own will , to the shock of the Fairy Tail audience . Sting decides to finish the Fairy Tail Dragon Slayers alone , for he held a promise to his Exceed friend Lector many years ago . As the three mages fight , the tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel , resulting in the collapse of the arena floor , and after a hand - to - hand combat underground , Sting was seemingly the victor of the match . However , the Fairy Tail duo stands up , comments on Sting 's strength , and argues about his position while attacking , thereby causing Natsu to drag Gajeel on a mining cart and send it away . While Gajeel is left incapacitated due to motion sickness , Natsu challenges the Twin Dragons of Saber Tooth alone , in retribution to what Sting did earlier on them . 175 `` Natsu vs. the Twin Dragons '' `` Natsu vs. Sōryū '' ( ナツ vs. 双竜 ) March 30 , 2013 September 1 , 2015 The tag battle of the Dragon Slayers comes to a close , with Natsu overpowering the combined attacks of Sting and Rogue . The Saber Tooth members watch in dismay , while Jiemma seethes in anger over the defeat of the Twin Dragons , frightening the nearby spectators away . The rest of Fairy Tail celebrates in joy as its team receives 10 points , finally surpassing Team Saber Tooth by a single point . Meanwhile , Gajeel , back to his senses after being dragged by Natsu into a cart earlier , stumbles upon a cave full of dragon skeletons . During that night , Jellal manages to corner the source of the dark aura , standing in disbelief at the identity of the mysterious , hooded person .
  Fact 0.0: Episode 151, 'Sabertooth', has the Japanese title 'Seibatusu' and originally aired on October 6, 2012.
  Fact 0.1: Episode 151 aired in English on November 17, 2015.
  Fact 0.2: The Legion Platoon is tasked by Zentopia's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again.
  Fact 0.3: Before embarking on their mission, the Legion Platoon visits Fairy Tail.
  Fact 0.4: The Legion Platoon makes amends with Fairy Tail for their previous hostility against them.
  Fact 0.5: It is later discovered that Fairy Tail has lost its title as the strongest guild in Fiore to Saber Tooth over the past seven years.
  Fact 0.6: Sting and Rogue are two of the strongest wizards in Fairy Tail.
  Fact 0.7: Sting and Rogue are a pair of Dragon Slayers.
  Fact 0.8: Sting is obsessed with fighting Natsu.
  Fact 0.9: Rogue is obsessed with fighting Gajeel.
  Fact 0.10: Makarov brings Gildarts to their abandoned guildhall.
  Fact 0.11: Makarov shows Gildarts Lumen Histoire, a light that Makarov says is Fairy Tail's greatest secret.
  Fact 0.12: Makarov tells Gildarts that he has been chosen to become the next master of Fairy Tail.
  Fact 0.13: Episode 152, 'And So We Aim for the Top', has the Japanese title 'Soshite Ore-tachi wa Chojo o Mezasu' and originally aired on October 13, 2012.
  Fact 0.14: Episode 152 aired in English on November 24, 2015.
  Fact 0.15: Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island.
  Fact 0.16: The group meets with Poluchka for medicine to strengthen their magic.
  Fact 0.17: The group is unsuccessful in strengthening their magic with the medicine.
  Fact 0.18: Wendy recognizes Poluchka's voice and scent as those of Grandine, the dragon who raised her.
  Fact 0.19: Poluchka reveals herself as the Edolas counterpart of Grandine.
  Fact 0.20: Poluchka gives Wendy written instructions for two advanced spells that Grandine failed to teach her.
  Fact 0.21: Makarov announces Gildarts' succession as master to the rest of the guild.
  Fact 0.22: Makarov finds that Gildarts has gone on a journey, leaving a letter of resignation behind.
  Fact 0.23: In the letter, Gildarts reinstates Laxus into the guild.
  Fact 0.24: In the letter, Gildarts returns mastership to Makarov.
  Fact 0.25: In the letter, Gildarts tells Makarov to make Fairy Tail the strongest guild in Fiore again.
  Fact 0.26: Romeo suggests they participate in the Grand Magic Games.
  Fact 0.27: The Grand Magic Games are an annual festival designed to determine the guild rankings in Fiore.
  Fact 0.28: Episode 153, 'Song of the Stars', has the Japanese title 'Hoshiboshi no Uta' and originally aired on October 20, 2012.
  Fact 0.29: Episode 153 aired in English on December 1, 2015.
  Fact 0.30: The Grand Magic Games are occurring in three months' time.
  Fact 0.31: Natsu and his friends train at a beach resort.
  Fact 0.32: On their second day of training, Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction.
  Fact 0.33: When the wizards arrive, they find a surprise party held by the Celestial Spirit King.
  Fact 0.34: The party celebrates the wizards' return from Sirius Island after seven years.
  Fact 0.35: The wizards spend the day in the spirit world partying with the spirits.
  Fact 0.36: Virgo informs the wizards that one day in the spirit world equals three months in the human world.
  Fact 0.37: The wizards find that the Grand Magic Games are upon them.
  Fact 0.38: Episode 154, 'For All the Time We Missed Each Other', has the Japanese title 'Surechigatta Jikan no Bun dake' and originally aired on October 27, 2012.
  Fact 0.39: Episode 154 aired in English on December 8, 2015.
  Fact 0.40: While mulling over their lost training time, the Fairy Tail wizards receive a message via carrier pigeon.
  Fact 0.41: The message requests a private meeting in the forest nearby.
  Fact 0.42: At the meeting, the wizards meet a runaway Jellal, Ultear, and Merudy.
  Fact 0.43: Jellal, Ultear, and Merudy have formed their own guild, Crime Sorciere.
  Fact 0.44: Crime Sorciere aims to defeat Zeref and any dark guilds that try to use his power.
  Fact 0.45: Jellal, Ultear, and Merudy ask Fairy Tail for help in determining the source of a magical anomaly similar to Zeref's magic.
  Fact 0.46: The magical anomaly has been sensed around the Grand Magic Games arena, Domus Flau.
  Fact 0.47: In exchange, Ultear uses her magic to invoke Fairy Tail's Second Origin in order to double their magic power.
  Fact 0.48: Erza talks with Jellal over his regained memories.
  Fact 0.49: Erza and Jellal nearly kiss.
  Fact 0.50: Jellal decides that he and Erza can not be together.
  Fact 0.51: Jellal lies to Erza that he has a fiancee.
  Fact 0.52: Erza quickly picks up on the lie and respects his decision.
  Fact 0.53: Erza sees Crime Sorciere off as they leave.
  Fact 0.54: Episode 155, 'Crocus, the Flower-Blooming Capital', has the Japanese title 'Hanasaku Miyako: Kurokkasu' and originally aired on November 3, 2012.
  Fact 0.55: Episode 155 aired in English on December 15, 2015.
  Fact 0.56: Natsu, Lucy, Gray, Erza, and Wendy are chosen to represent their guild in the Grand Magic Games.
  Fact 0.57: The Grand Magic Games are to be held in Fiore's capital, Crocus.
  Fact 0.58: As they tour the city, Natsu and Happy encounter the twin dragon slayers, Sting and Rogue.
  Fact 0.59: Sting and Rogue mock Natsu for not being able to kill Acnologia.
  Fact 0.60: Sting and Rogue boast of becoming true dragon slayers by killing the dragons who raised them.
  Fact 0.61: At midnight, Fairy Tail and the other participating guilds are put through a preliminary event.
  Fact 0.62: The preliminary event is a race through a massive floating labyrinth.
  Fact 0.63: The preliminary event is designed to narrow the number of participating guilds down to eight teams.
  Fact 0.64: Episode 156, 'Sky Labyrinth', has the Japanese title 'Sukai Rabirinsu' and originally aired on November 10, 2012.
  Fact 0.65: Episode 156 aired in English on December 22, 2015.
  Fact 0.66: Natsu's team notices that Wendy has gone missing.
  Fact 0.67: Elfman replaces Wendy to keep Natsu's team from being eliminated.
  Fact 0.68: As they make their way through the enchanted labyrinth, the five encounter various other guilds.
  Fact 0.69: The five steal the maps made by the other guilds to better navigate the maze.
  Fact 0.70: The five reach the end of the maze and pass the elimination round.
  Fact 0.71: The five are confident they have arrived first.
  Fact 0.72: The five discover they have barely taken last place.
  Fact 0.73: Happy, Lisanna, and the rest of the guild search for Wendy and Carla.
  Fact 0.74: Happy, Lisanna, and the rest of the guild find Wendy and Carla in the garden of the king's palace, Mercurius.
  Fact 0.75: Wendy and Carla have their magic power depleted.
  Fact 0.76: Episode 157, 'New Guild', has the Japanese title 'Shinki Girudo' and originally aired on November 17, 2012.
  Fact 0.77: Episode 157 aired in English on December 29, 2015.
  Fact 0.78: Wendy's body is weakened as a result of losing her magic power.
  Fact 0.79: Poluchka helps Wendy recover.
  Fact 0.80: Wendy formally requests Elfman to take her place in the Grand Magic Games.
  Fact 0.81: Fairy Tail steps into the arena with a negative audience reception.
  Fact 0.82: Fairy Tail is bolstered by their guild mates and the spirit of Mavis, whom only Fairy Tail guild members can see.
  Fact 0.83: The next four participating teams are Quatro Cerberus, Mermaid Heel, Blue Pegasus, and Lamia Scale.
  Fact 0.84: Raven Tail is a former dark guild led by Makarov's son Ivan.
  Fact 0.85: Raven Tail members confess to attacking Wendy and Carla.
  Fact 0.86: The next team is a second team from Fairy Tail composed of Gajeel, Juvia, Mirajane, Laxus, and Jellal.
  Fact 0.87: Jellal is disguised as Mystogan to help investigate the arena.
  Fact 0.88: The final guild to be announced is Saber Tooth.
  Fact 0.89: The audience cheers joyfully for Saber Tooth's announcement.
  Fact 0.90: All eight teams are announced.
  Fact 0.91: Each team sends out one member to participate in the first contest of the games.
  Fact 0.92: Gray and Juvia represent their teams in the first contest.
  Fact 0.93: Episode 158, 'Night of Shooting Stars', has the Japanese title 'Hoshi Furu Yoru ni' and originally aired on November 24, 2012.
  Fact 0.94: Episode 158 aired in English on January 5, 2016.
  Fact 0.95: The games' first contest, 'Hidden', requires the eight participants to score points by finding and attacking each other.
  Fact 0.96: The contest takes place in a virtual city filled with copies of the participants.
  Fact 0.97: Raven Tail member Nalpudding repeatedly targets Gray.
  Fact 0.98: Gray scores last place in the contest.
  Fact 0.99: Saber Tooth's Rufus scores first by using his Memory-Make magic to attack all the other participants at once.
  Fact 0.100: Gray is humiliated and swears revenge against Raven Tail and Saber Tooth.
  Fact 0.101: Lucy is chosen to fight Raven Tail member Flare in the tournament's first battle round.
  Fact 0.102: Episode 159, 'Lucy vs. Flare', has the Japanese title 'Rushii vs. Furea' and originally aired on December 1, 2012.
  Fact 0.103: Episode 159 aired in English on January 12, 2016.
  Fact 0.104: Lucy displays tremendous skill in her battle against Flare by summoning two spirits simultaneously.
  Fact 0.105: Flare cheats by using her magic hair to hold Alzack and Bisca's daughter, Asuka, as a hostage.
  Fact 0.106: Flare blackmails Lucy into not fighting back to keep Flare from harming Asuka.
  Fact 0.107: Lucy tearfully speaks of Asuka's danger.
  Fact 0.108: Natsu hears Lucy and rushes to Asuka.
  Fact 0.109: Natsu incinerates Flare's hair.
  Fact 0.110: Lucy counterattacks with her Urano Metoria spell in combination with Gemini.
  Fact 0.111: Flare's teammate Obra nullifies Lucy's magic before the spell fully activates.
  Fact 0.112: Lucy collapses and loses the match.
  Fact 0.113: Episode 160, 'Portent', has the Japanese title 'Kyozui' and originally aired on December 8, 2012.
  Fact 0.114: Episode 160 aired in English on January 19, 2016.
  Fact 0.115: Blue Pegasus' Ren defeats Mermaid Heel's Arana during the games' second match.
  Fact 0.116: Saber Tooth's Orga effortlessly defeats Quatro Cerberus' War Cry in the third match.
  Fact 0.117: For the day's fourth and final match, Jellal is pitted against Lamia Scale's Jura.
  Fact 0.118: Jellal intends to avenge the humiliated Fairy Tail members.
  Fact 0.119: Jellal mimics Mystogan's magic to keep up his disguise.
  Fact 0.120: Jellal finds Mystogan's magic ineffective against Jura.
  Fact 0.121: Jellal is forced to use his own magic instead.
  Fact 0.122: Before Jellal can cast a spell that will expose his true identity, Ultear uses Merudy's sensory to force Jellal into submission.
  Fact 0.123: The hospitalized Carla awakens after having a troubling premonition of the future.
  Fact 0.124: Episode 161, 'Chariots', has the Japanese title 'Chariotto' and originally aired on December 15, 2012.
  Fact 0.125: Episode 161 aired in English on January 26, 2016.
  Fact 0.126: Fairy Tail is undaunted by their poor performance on the first day.
  Fact 0.127: As the guild celebrates, they are visited and taunted by Bacchus.
  Fact 0.128: Bacchus is Erza's rival from Quatro Cerberus.
  Fact 0.129: Bacchus has come to replace War Cry.
  Fact 0.130: Carla relays her premonition to Poluchka.
  Fact 0.131: In Carla's premonition, Mercurius is collapsing around what appeared to be a tearfully singing Lucy.
  Fact 0.132: The second day of the tournament begins with a footrace atop a train of moving chariots.
  Fact 0.133: Natsu, Gajeel, and Sting suffer from motion sickness as a side effect of their Dragon Slayer magic.
  Fact 0.134: Natsu responds to Sting that his determination is a result of the strength of Fairy Tail's bonds.
  Fact 0.135: Sting eventually throws the race.
  Fact 0.136: Natsu finishes as sixth place.
  Fact 0.137: Gajeel finishes as seventh place.
  Fact 0.138: Natsu and Gajeel earn Fairy Tail the audience's respect.
  Fact 0.139: Episode 162, 'Elfman vs. Bacchus', has the Japanese title 'Erufuman vs. Bakkasu' and originally aired on December 22, 2012.
  Fact 0.140: Episode 162 aired in English on September 1, 2015.
  Fact 0.141: Due to a miscommunication between the king and his knight Arcadios, Elfman is chosen as Bacchus' opponent instead of Erza.
  Fact 0.142: Before the match, Bacchus and Elfman make a wager.
  Fact 0.143: If Bacchus wins, he will court Elfman's sisters.
  Fact 0.144: If Elfman wins, Bacchus' team name will be changed to 'Quatro Puppy' for the rest of the games.
  Fact 0.145: Bacchus ruthlessly attacks Elfman with his bare palms.
  Fact 0.146: Elfman damages Bacchus with his defensive Lizardman transformation.
  Fact 0.147: Elfman outlasts Bacchus as Bacchus exhausts himself.
  Fact 0.148: Elfman wins the match.
  Fact 0.149: Bacchus' team is renamed.
  Fact 0.150: Natsu finds Wendy, Carla, and Poluchka being kidnapped by a group of masked men.
  Fact 0.151: The masked men claim they were hired to kidnap Lucy by Raven Tail.
  Fact 0.152: The masked men are revealed to have actually been sent by Arcadios.
  Fact 0.153: Arcadios intends to sow discord between the guilds.
  Fact 0.154: Arcadios plots to use Lucy for his 'Eclipse Project'.
  Fact 0.155: Episode 163, 'Mirajane vs. Jenny', has the Japanese title 'Mirajen vs. Jeni' and originally aired on January 5, 2013.
  Fact 0.156: Episode 163 aired in English on September 1, 2015.
  Fact 0.157: The second day of the Grand Magic Games continues.
  Fact 0.158: Mirajane of Fairy Tail Team B fights against Jenny Realight of Team Blue Pegasus.
  Fact 0.159: Mirajane and Jenny are well-known for being magazine models.
  Fact 0.160: Their battle consists almost entirely of a fashion showdown.
  Fact 0.161: Jenny proposes a wager that the loser must pose nude for the Sorcerer Magazine.
  Fact 0.162: Mirajane accepts the wager.
  Fact 0.163: Jenny transforms into her battle armor.
  Fact 0.164: Mirajane shifts into her Satan Soul 'Sitri' form.
  Fact 0.165: Mirajane easily defeats Jenny.
  Fact 0.166: The final match of the day is Kagura Mikazuchi of Mermaid Heel against Yukino Aguria of Saber Tooth.
  Fact 0.167: Yukino proposes to Kagura that they wager their lives on their battle.
  Fact 0.168: Arcadios is confronted by Darton, the kingdom's defense minister, about his 'Eclipse Project'.
  Fact 0.169: Episode 164, 'Kagura vs. Yukino', has the Japanese title 'Kagura vs. Yukino' and originally aired on January 12, 2013.
  Fact 0.170: Episode 164 aired in English on September 1, 2015.
  Fact 0.171: Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura.
  Fact 0.172: Kagura defeats Libra and Pisces using her gravity-altering magic.
  Fact 0.173: Yukino uses a Gatekey that summons Ophiuchus, which is regarded as the strongest among the Celestial Spirits.
  Fact 0.174: Kagura slays Ophiuchus with her sword still sheathed.
  Fact 0.175: Kagura easily defeats Yukino.
  Fact 0.176: Jellal discusses the fact that he has not sensed the evil power similar to Zeref's.
  Fact 0.177: Jellal's discussion takes place after the second day of the Grand Magic Games has ended.
  Fact 0.178: Erza reunites with Milliana.
  Fact 0.179: Milliana is revealed to be a member of Mermaid Heel.
  Fact 0.180: Saber Tooth's master Jiemma is dissatisfied with his guild's performance.
  Fact 0.181: Sting is given another chance.
  Fact 0.182: Jiemma humiliates Yukino for her failure.
  Fact 0.183: Episode 165, 'Hatred at Nightfall', has the Japanese title 'Urami wa Yoru no Tobari ni Tsutsumarete' and originally aired on January 19, 2013.
  Fact 0.184: Episode 165 aired in English on September 1, 2015.
  Fact 0.185: After Yukino is expelled from Saber Tooth, she offers her two golden Gatekeys to Lucy.
  Fact 0.186: Yukino offers the Gatekeys to complete Lucy's collection of the Zodiac Gatekeys.
  Fact 0.187: Yukino states that when the twelve Zodiac Gatekeys are all gathered, a gateway that changes the world will open.
  Fact 0.188: Lucy turns down the offer.
  Fact 0.189: Milliana tells Erza that Kagura's sword, Archenemy, is intended to kill Jellal.
  Fact 0.190: Milliana tells Erza that she and Kagura share the same resentment towards Jellal.
  Fact 0.191: Natsu and Happy run after Yukino in an attempt to apologize.
  Fact 0.192: Natsu realizes that Yukino had been expelled.
  Fact 0.193: Natsu infiltrates the Crocus Garden to fight Jiemma.
  Fact 0.194: Episode 166, 'Pandemonium', has the Japanese title 'Pandemoniumu' and originally aired on January 26, 2013.
  Fact 0.195: Episode 166 aired in English on September 1, 2015.
  Fact 0.196: Natsu punches out Dobengal, the guild member summoned to fight him.
  Fact 0.197: Natsu and Jiemma engage in a duel.
  Fact 0.198: The guild master's daughter, Minerva, appears and ends their duel.
  Fact 0.199: Minerva takes Happy hostage.
  Fact 0.200: Minerva convinces Natsu to settle their dispute in the Grand Magic Games.
  Fact 0.201: As Natsu leaves, he states that Saber Tooth will never surpass Fairy Tail because they care for their friends.
  Fact 0.202: Rogue echoes Natsu's words.
  Fact 0.203: Sting is left in awe at Natsu's strength.
  Fact 0.204: The following day, an event called 'Pandemonium' begins.
  Fact 0.205: In Pandemonium, participants enter a magical citadel filled with 100 monsters of varying strengths.
  Fact 0.206: Participants individually do battle with any number of monsters they choose.
  Fact 0.207: Erza, representing Fairy Tail Team A, requests the 100 monsters as her opponents.
  Fact 0.208: Episode 167, '100 Against 1', has the Japanese title '100 tai 1' and originally aired on February 2, 2013.
  Fact 0.209: Episode 167 aired in English on September 1, 2015.
  Fact 0.210: Erza kills the 100 monsters using various types of armor to match their strengths and weaknesses.
  Fact 0.211: The crowd praises Fairy Tail.
  Fact 0.212: Fairy Tail is awarded 10 points.
  Fact 0.213: The game officials decide to determine the ranks of the remaining participants by means of a device called the 'Magic Power Finder', sometimes abbreviated as MPF.
  Fact 0.214: Participants must attack the Magic Power Finder with their magic power.
  Fact 0.215: The Magic Power Finder calculates and displays participants' equivalent scores based on the extent of power used.
  Fact 0.216: Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis.
  Fact 0.217: Makarov is surprised by Cana's use of the Fairy Glitter spell.
  Fact 0.218: Fairy Tail Team B receives 8 points.
  Fact 0.219: The audience bursts into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams.
  Fact 0.220: The masked leader Alexei tells his subordinates Flare, Obra, Nalpudding, and Kurohebi that it is about time to proceed with their true mission.
  Fact 0.221: Raven Tail somehow got insider information regarding the battle matchups for the day.
  Fact 0.222: Episode 168, 'Laxus vs. Alexei', has the Japanese title 'Rakusasu vs. Arekusei' and originally aired on February 9, 2013.
  Fact 0.223: Episode 168 aired in English on September 1, 2015.
  Fact 0.224: In the battle portion of the Grand Magic Games' third day, Milliana of Mermaid Heel beats Semmes of Quatro Puppy.
  Fact 0.225: Saber Tooth's Rufus swiftly defeats Blue Pegasus' Eve.
  Fact 0.226: Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail.
  Fact 0.227: The Fairy Tail members take extra precautions because Raven Tail is a suspicious guild.
  Fact 0.228: Fairy Tail watches the actions of Ivan Dreyar and the other four Raven Tail members.
  Fact 0.229: The spectators are shocked to see Laxus being beaten by Alexei.
  Fact 0.230: No cautious movements can be seen from the Raven Tail team.
  Fact 0.231: The two dueling people are illusions created by Alexei.
  Fact 0.232: Alexei created the illusions so that he could have a talk with Laxus.
  Fact 0.233: The other Raven Tail members being monitored by Fairy Tail are lifelike illusions.
  Fact 0.234: In reality, the other Raven Tail members are with Alexei on the battlefield.
  Fact 0.235: Alexei reveals himself to be Ivan Dreyar.
  Fact 0.236: Ivan and the Raven Tail members are unable to interrogate Laxus about the Lumen Histoire's location.
  Fact 0.237: Ivan and the Raven Tail members are forced to fight Laxus.
  Fact 0.238: Before their battle begins, Ivan declares that Raven Tail is an anti-Fairy Tail guild.
  Fact 0.239: Episode 169, 'Wendy vs. Shelia', has the Japanese title 'Wendi vs. Sheria' and originally aired on February 16, 2013.
  Fact 0.240: Episode 169 aired in English on September 1, 2015.
  Fact 0.241: Laxus defeats Ivan and the four Raven Tail elite members.
  Fact 0.242: Ivan's illusions disappear to the shock of everyone in the stadium.
  Fact 0.243: The game officials declare Laxus the victor.
  Fact 0.244: Ivan tells Laxus that Lumen Histoire is Fairy Tail's 'darkness'.
  Fact 0.245: Raven Tail is disqualified for the remainder of the Games due to the infraction of the rules.
  Fact 0.246: Raven Tail's members are captured by the soldiers.
  Fact 0.247: Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off for the final match of the day.
  Fact 0.248: Sherria is revealed to be a Sky God Slayer.
  Fact 0.249: Gusts of wind surround the arena as Sherria and Wendy exchange attacks.
  Fact 0.250: Wendy employs an advanced Sky Dragon Slayer attack, much to Poluchka's astonishment.
  Fact 0.251: Sherria falls on the ground.
  Fact 0.252: Wendy is about to be announced as the winner of the bout.
  Fact 0.253: Sherria suddenly stands up with her wounds fully healed.
  Fact 0.254: The Crime Sorciere guild senses the nefarious aura.
  Fact 0.255: Jellal proceeds to the Domus Flau audience.
  Fact 0.256: Jellal is oblivious that he bumped into Doranbolt.
  Fact 0.257: Doranbolt wonders what Mystogan is doing in Earthland.
  Fact 0.258: Episode 170, 'Small Fists', has the Japanese title 'Chiisana Kobushi' and originally aired on February 23, 2013.
  Fact 0.259: Episode 170 aired in English on September 1, 2015.
  Fact 0.260: Cheria is fully healed.
  Fact 0.261: Wendy is almost depleted of magic power.
  Fact 0.262: Cheria casts a high-level spell towards Wendy.
  Fact 0.263: Wendy enhances Cheria's physical strength, causing the attack to miss Wendy.
  Fact 0.264: The two competitors engage in hand-to-hand combat.
  Fact 0.265: Due to the time limit and neither side giving up, their battle results in a draw.
  Fact 0.266: The draw gives both teams 5 points.
  Fact 0.267: Jellal tracks the sinister aura among the crowd in the form of a mysterious, hooded person.
  Fact 0.268: Jellal is about to confront the aura source.
  Fact 0.269: Doranbolt, Lahar, and other Rune Knights corner Jellal.
  Fact 0.270: Doranbolt interrogates Jellal.
  Fact 0.271: Jellal's cover is blown.
  Fact 0.272: The Rune Knights, Kagura and Milliana see the true identity of Mystogan.
  Fact 0.273: Yajima convinces the Rune Knights that Mystogan is Jellal's Edolas counterpart.
  Fact 0.274: Lahar releases the fugitive.
  Fact 0.275: Laxus and Makarov have a small conversation regarding Lumen Histoire.
  Fact 0.276: Mavis appears and explains that Lumen Histoire is the 'light' of Fairy Tail.
  Fact 0.277: Episode 171, 'Naval Battle', has the Japanese title 'Nabaru Batoru' and originally aired on March 2, 2013.
  Fact 0.278: Episode 171 aired in English on September 1, 2015.
  Fact 0.279: The jubilant Fairy Tail members celebrate their victories.
  Fact 0.280: Sting, Kagura, and Milliana are contemplating their own concerns.
  Fact 0.281: Yukino is enlisted as a sergeant under Arcadios' command.
  Fact 0.282: The fourth day of the competitions begins with an event called 'Naval Battle'.
  Fact 0.283: Naval Battle involves participants swimming in a giant water sphere and knocking the other competitors out.
  Fact 0.284: After the other participants fall out of the sphere, a five-minute battle between Lucy and Minerva ensues.
  Fact 0.285: Minerva quickly attacks Lucy and steals the keys of the Celestial Spirit mage.
  Fact 0.286: Lucy refuses to surrender.
  Fact 0.287: After hearing Lucy's intentions, Minerva stops attacking until the five-minute mark is reached.
  Fact 0.288: Minerva resumes attacking Lucy until the game officials decide to halt the match.
  Fact 0.289: The Fairy Tail members snap.
  Fact 0.290: Minerva dangles Lucy out of the sphere.
  Fact 0.291: The other Fairy Tail members rush to Lucy's aid.
  Fact 0.292: Episode 172, 'A Parfum for You', has the Japanese title 'Kimi ni Sasageru Parufamu' and originally aired on March 9, 2013.
  Fact 0.293: Episode 172 aired in English on September 1, 2015.
  Fact 0.294: Lucy is hospitalized and taken to the infirmary.
  Fact 0.295: The Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so.
  Fact 0.296: Makarov announces that the two Fairy Tail teams must merge.
  Fact 0.297: The merger is a result of the disqualification of Raven Tail.
  Fact 0.298: The merger compensates for the odd number of teams for the battle round.
  Fact 0.299: The new team will have the former Team A's 35 points to start with.
  Fact 0.300: The new Fairy Tail team consists of Natsu, Gray, Erza, Gajeel, and Laxus.
  Fact 0.301: The new team is well-received as they enter the field.
  Fact 0.302: The second part of the tournament begins with a tag battle between Blue Pegasus' Ichiya and 'Rabbit' and Quatro Puppy's Rocker and Bacchus.
  Fact 0.303: 'Rabbit' is revealed to be Ichiya in disguise.
  Fact 0.304: Bacchus punches the Exceed.
  Fact 0.305: Ichiya bulks up to defeat the Quatro Puppy opponents.
  Fact 0.306: Blue Pegasus is awarded 10 points.
  Fact 0.307: Episode 173, 'Battle of Dragon Slayers!', has the Japanese title 'Batoru obu Doragon Sureiya' and originally aired on March 16, 2013.
  Fact 0.308: Episode 173 aired in English on September 1, 2015.
  Fact 0.309: Mermaid Heel's Kagura and Milliana face off against Lamia Scale's Lyon and Yuka for the second tag battle of the games.
  Fact 0.310: The second tag battle ends in a draw due to the time limit.
  Fact 0.311: For the third and final match of the day, Natsu and Gajeel of Team Fairy Tail enter the field against Sting and Rogue of Team Saber Tooth.
  Fact 0.312: Sting and Rogue were about to take the initiative when Natsu and Gajeel attacked them.
  Fact 0.313: The audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers.
  Fact 0.314: Sting activates 'White Drive'.
  Fact 0.315: Rogue powers up using 'Shadow Drive'.
  Fact 0.316: Episode 174, 'Four Dragons', has the Japanese title 'Yonin no Doragon' and originally aired on March 23, 2013.
  Fact 0.317: Episode 174 aired in English on September 1, 2015.
  Fact 0.318: Sting and Rogue begin attacking Natsu and Gajeel with their powers increased.
  Fact 0.319: After receiving heavy blows from the Saber Tooth Dragon Slayers, Natsu and Gajeel manage to hold them off.
  Fact 0.320: Sting and Rogue are forced to activate 'Dragon Force' on their own will.
  Fact 0.321: The Fairy Tail audience is shocked by Sting and Rogue activating 'Dragon Force'.
  Fact 0.322: Sting decides to finish the Fairy Tail Dragon Slayers alone.
  Fact 0.323: Sting held a promise to his Exceed friend Lector many years ago.
  Fact 0.324: The tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel.
  Fact 0.325: Sting's attack causes the collapse of the arena floor.
  Fact 0.326: After a hand-to-hand combat underground, Sting is seemingly the victor of the match.
  Fact 0.327: Natsu and Gajeel stand up, comment on Sting's strength, and argue about his position while attacking.
  Fact 0.328: Natsu drags Gajeel on a mining cart and sends it away.
  Fact 0.329: Gajeel is left incapacitated due to motion sickness.
  Fact 0.330: Natsu challenges the Twin Dragons of Saber Tooth alone, in retribution for what Sting did earlier.
  Fact 0.331: Episode 175, 'Natsu vs. the Twin Dragons', has the Japanese title 'Natsu vs. Soryu' and originally aired on March 30, 2013.
  Fact 0.332: Episode 175 aired in English on September 1, 2015.
  Fact 0.333: The tag battle of the Dragon Slayers comes to a close.
  Fact 0.334: Natsu overpowers the combined attacks of Sting and Rogue.
  Fact 0.335: The Saber Tooth members watch in dismay.
  Fact 0.336: Jiemma seethes in anger over the defeat of the Twin Dragons.
  Fact 0.337: Jiemma's anger frightens the nearby spectators away.
  Fact 0.338: The rest of Fairy Tail celebrates in joy as its team receives 10 points.
  Fact 0.339: Fairy Tail surpasses Team Saber Tooth by a single point.
  Fact 0.340: Gajeel, back to his senses after being dragged by Natsu into a cart earlier, stumbles upon a cave full of dragon skeletons.
  Fact 0.341: That night, Jellal manages to corner the source of the dark aura.
  Fact 0.342: Jellal stands in disbelief at the identity of the mysterious, hooded person.

Retrieved items:
<item 1>
List of Fairy Tail episodes (season 6)
Meanwhile , Milliana tells Erza that Kagura 's sword , Archenemy , is intended to kill Jellal , and that she and Kagura share the same resentment towards Jellal
. Natsu and Happy run after Yukino in an attempt to apologize ; however , Natsu realizes that Yukino had been expelled , and infiltrates the Crocus Garden to
fight Jiemma . 166 `` Pandemonium '' `` Pandemoniumu '' ( 伏魔殿 ( パンデモニウム ) ) January 26 , 2013 September 1 , 2015 After Natsu punches out Dobengal , the guild
member summoned to fight him , he and Jiemma engage in a duel . However , the guild master 's daughter Minerva appears , ends their duel and , taking Happy
hostage , convinces Natsu to settle their dispute in the Grand Magic Games . As Natsu leaves , he states that Saber Tooth will never surpass Fairy Tail because
they care for their friends , words that are echoed by Rogue , while Sting is left in awe at Natsu 's strength .
</item 1>

<item 2>
List of Fairy Tail episodes (season 6)
In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's
victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the
two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked
leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei
'' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of
Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve .
</item 2>

<item 3>
List of Fairy Tail episodes (season 6)
When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years .
They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals
three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no
Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message
via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild
, Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power .
</item 3>

<item 4>
List of Fairy Tail episodes (season 6)
Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace ,
Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is
weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy
Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail
guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is
Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla .
</item 4>

<item 5>
List of Fairy Tail episodes (season 6)
It contains a single story arc , titled `` Dai Matō Enbu '' ( 大 魔 闘 演武 , lit `` Grand Magic Games '' ) , which adapts material from the beginning of the 31st
through the ending of the 35th volumes of Hiro Mashima 's Fairy Tail manga over 24 episodes . Focusing on Natsu and the others who were frozen in time for seven
years on Tenrou Island , the members return to Magnolia and discover that Saber Tooth is the strongest guild in the Fiore Kingdom . Having become the weakest
guild during the absence of its core members , Fairy Tail decides to enter the Grand Magic Games , an annual competition to decide Fiore 's strongest guild .
The season began airing on October 6 , 2012 on TV Tokyo in Japan and concluded its run on March 30 , 2013 . The first DVD compilation , containing the last two
episodes of the fifth season and the first two episodes of the sixth season , was released by Pony Canyon on March 6 , 2013 .
</item 5>

<item 6>
List of Fairy Tail episodes (season 6)
Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza
reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance
. Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete ''
( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete
the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ;
however , Lucy turns down the offer .
</item 6>

<item 7>
List of Fairy Tail episodes (season 6)
Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her
. Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of
resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest
guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 ``
Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu
and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of
destruction .
</item 7>

<item 8>
List of Fairy Tail episodes (season 6)
The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled
with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests
the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters
using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points .
Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes
abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent
scores depending on the extent of power used .
</item 8>

<item 9>
List of Fairy Tail episodes (season 6)
Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy
by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his ``
Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand
Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for
being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes
redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts .
</item 9>

<item 10>
List of Fairy Tail episodes (season 6)
The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic
Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the
painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza
can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave .
155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray
, Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus .
</item 10>
=== END ITEM id=096577dd00eb9cad ===

=== ITEM id=45b612f12c8daa6b ===
Gold sentences and facts:
Sentence 0: ESPN Monday Night Football ( abbreviated as MNF and also known as ESPN Monday Night Football on ABC for rare live special broadcast ) is a live television broadcast of weekly National Football League ( NFL ) games on ESPN in the United States . From 1970 to 2005 , it aired on sister broadcast network ABC . Monday Night Football was , along with Hallmark Hall of Fame and the Walt Disney anthology television series , one of the longest - running prime time programs ever on commercial network television , and one of the highest - rated , particularly among male viewers . MNF is preceded on air by Monday Night Countdown served by Applebee 's .
  Fact 0.0: ESPN Monday Night Football is abbreviated as MNF.
  Fact 0.1: ESPN Monday Night Football is also known as ESPN Monday Night Football on ABC for rare live special broadcast.
  Fact 0.2: ESPN Monday Night Football is a live television broadcast of weekly National Football League (NFL) games.
  Fact 0.3: ESPN Monday Night Football airs on ESPN in the United States.
  Fact 0.4: From 1970 to 2005, Monday Night Football aired on sister broadcast network ABC.
  Fact 0.5: Monday Night Football is one of the longest-running prime time programs ever on commercial network television.
  Fact 0.6: Monday Night Football is, along with Hallmark Hall of Fame and the Walt Disney anthology television series, one of the longest-running prime time programs ever on commercial network television.
  Fact 0.7: Monday Night Football is one of the highest-rated programs, particularly among male viewers.
  Fact 0.8: Monday Night Football is preceded on air by Monday Night Countdown.
  Fact 0.9: Monday Night Countdown is served by Applebee's.
Sentence 1: ESPN Monday Night Football Genre NFL game telecasts Created by Roone Arledge Directed by Chet Forte ( 1970 -- 1986 ) Larry Kamm ( 1987 ) Craig Janoff ( 1988 -- 1999 ) Drew Esocoff ( 2000 -- 2005 ) Chip Dean ( 2006 -- present ) Presented by Commentators : Sean McDonough Lisa Salters Studio hosts : Suzy Kolber Steve Young Randy Moss Matt Hasselbeck Charles Woodson Opening theme `` Score '' by Bob 's Band ( 1970 -- 1975 ) `` All My Rowdy Friends Are Here on Monday Night '' by Hank Williams , Jr. ( 1989 -- 2011 ; 2017 -- present ) `` Heavy Action '' by Johnny Pearson ( 1976 -- 1988 ; 2011 -- 2016 ) ( used as secondary theme from 1989 to 2011 and again from 2017 -- present ) Country of origin United States Original language ( s ) English No. of seasons 47 ( NFL seasons ) No. of episodes 718 ( games ) Production Executive producer ( s ) Norby Williamson ( 2006 -- present ) Producer ( s ) Roger Lewin Jay Rothman Suzy Kolber ( uncredited ; 2006 -- present ) Location ( s ) Various NFL stadiums ( game telecasts ) ESPN Center , Bristol , Connecticut ( studio segments ) Camera setup Multi-camera Running time 3 - 4 hours Production company ( s ) National Football League ABC Sports ( 1970 -- 2005 ) ESPN ( 2006 -- present ) Release Original network ABC ( 1970 -- 2005 ) ESPN ( 2006 -- present ) Picture format 480i ( SDTV ) ( 1970 -- 2010 ) , 480i ( 16 : 9 SDTV ) ( 2011 -- present ) , 720p ( HDTV ) ( 2011 -- present ) Original release September 21 , 1970 ( 1970 - 09 - 21 ) -- present Chronology Related shows Monday Night Countdown External links Website espn.go.com/nfl/mnf
  Fact 1.0: ESPN Monday Night Football has the genre NFL game telecasts.
  Fact 1.1: ESPN Monday Night Football was created by Roone Arledge.
  Fact 1.2: Chet Forte directed ESPN Monday Night Football from 1970 to 1986.
  Fact 1.3: Larry Kamm directed ESPN Monday Night Football in 1987.
  Fact 1.4: Craig Janoff directed ESPN Monday Night Football from 1988 to 1999.
  Fact 1.5: Drew Esocoff directed ESPN Monday Night Football from 2000 to 2005.
  Fact 1.6: Chip Dean has directed ESPN Monday Night Football from 2006 to the present.
  Fact 1.7: The commentators of ESPN Monday Night Football are Sean McDonough and Lisa Salters.
  Fact 1.8: The studio hosts of ESPN Monday Night Football are Suzy Kolber, Steve Young, Randy Moss, Matt Hasselbeck and Charles Woodson.
  Fact 1.9: The opening theme 'Score' by Bob's Band was used from 1970 to 1975.
  Fact 1.10: The opening theme 'All My Rowdy Friends Are Here on Monday Night' by Hank Williams, Jr. was used from 1989 to 2011 and again from 2017 to the present.
  Fact 1.11: The opening theme 'Heavy Action' by Johnny Pearson was used from 1976 to 1988 and from 2011 to 2016.
  Fact 1.12: 'Heavy Action' was used as the secondary theme from 1989 to 2011 and again from 2017 to the present.
  Fact 1.13: The country of origin of ESPN Monday Night Football is the United States.
  Fact 1.14: The original language of ESPN Monday Night Football is English.
  Fact 1.15: ESPN Monday Night Football has had 47 NFL seasons.
  Fact 1.16: ESPN Monday Night Football has had 718 games as episodes.
  Fact 1.17: Norby Williamson has been executive producer of ESPN Monday Night Football from 2006 to the present.
  Fact 1.18: Roger Lewin and Jay Rothman were producers of ESPN Monday Night Football.
  Fact 1.19: Suzy Kolber is an uncredited producer of ESPN Monday Night Football from 2006 to the present.
  Fact 1.20: ESPN Monday Night Football game telecasts are located at various NFL stadiums.
  Fact 1.21: ESPN Monday Night Football studio segments are produced at ESPN Center in Bristol, Connecticut.
  Fact 1.22: ESPN Monday Night Football uses a multi-camera setup.
  Fact 1.23: ESPN Monday Night Football has a running time of 3 to 4 hours.
  Fact 1.24: The National Football League is a production company of ESPN Monday Night Football.
  Fact 1.25: ABC Sports was a production company of ESPN Monday Night Football from 1970 to 2005.
  Fact 1.26: ESPN was a production company of ESPN Monday Night Football from 2006 to the present.
  Fact 1.27: The original network of ESPN Monday Night Football was ABC from 1970 to 2005.
  Fact 1.28: The original network of ESPN Monday Night Football is ESPN from 2006 to the present.
  Fact 1.29: ESPN Monday Night Football was broadcast in 480i (SDTV) picture format from 1970 to 2010.
  Fact 1.30: ESPN Monday Night Football was broadcast in 480i (16:9 SDTV) picture format from 2011 to the present.
  Fact 1.31: ESPN Monday Night Football was broadcast in 720p (HDTV) picture format from 2011 to the present.
  Fact 1.32: ESPN Monday Night Football originally released on September 21, 1970.
  Fact 1.33: ESPN Monday Night Football has been released from September 21, 1970 to the present.
  Fact 1.34: The related show of ESPN Monday Night Football is Monday Night Countdown.
  Fact 1.35: The website of ESPN Monday Night Football is espn.go.com/nfl/mnf.

Retrieved items:
<item 1>
Monday Night Football
As of 2008 , Westwood One was no longer controlled by CBS , but the network retained its NFL broadcast rights . In 2011 , Westwood One was purchased by Dial
Global . Then in 2014 , Dial Global , including Westwood One , was acquired by Cumulus Media . The Spanish - language broadcast is carried on ESPN Deportes
Radio . Nielsen ratings ( edit ) This article needs to be updated . Please update this article to reflect recent events or newly available information . (
December 2016 ) The highest - rated Monday Night Football telecast on ABC was the Miami Dolphins ' victory over the previously undefeated Chicago Bears on
December 2 , 1985 , which drew a national Nielsen rating of 29.6 and a share of 46 . ABC 's lowest - rated MNF game was the St. Louis Rams ' defeat of the Tampa
Bay Buccaneers on October 18 , 2004 , which drew a 7.7 rating . During the 1995 -- 1996 television season , Monday Night Football averaged a 17.1 household
rating . Its main competitor , Murphy Brown on CBS , averaged a 12.3 rating .
</item 1>

<item 2>
Monday Night Football
1987 -- 2005 ) NFL Draft NFL on ABC Sports Football Sunday on ESPN Radio Studio programs The Blitz Monday Night Countdown NFL Matchup NFL Primetime Sunday NFL
Countdown NFL Live NFL Insiders Other football programs Arena Football League on ESPN CFL on TSN ESPN College Football USFL on ESPN Related articles History of
Monday Night Football NFL on television ( history ) Commentators Sean McDonough Play - by - play TBA Color commentary Lisa Salters Sideline reporter Former
commentators Bonnie Bernstein Mike Ditka Jon Gruden Ron Jaworski Suzy Kolber Tony Kornheiser Michele Tafoya Joe Theismann Mike Tirico Dick Vermeil Steve Young
Lore televised by ESPN ESPN takes over MNF 2005 Saints -- Giants / Cowboys -- Redskins Doubleheader 2006 Bears -- Cardinals Comeback Fail Mary Sunday Night
Football era American Bowl ( 1990 - 2005 ) Fútbol Americano Pro Bowls broadcast on ESPN 1988 1990 1991 1992 1994 2005 2006 2015 2016 2017 2018 Music Edd
Kalehoff Johnny Pearson Hank Williams , Jr . Songs `` All My Rowdy Friends Are Here on Monday Night '' `` Heavy
</item 2>

<item 3>
Monday Night Football
Jr . Songs `` All My Rowdy Friends Are Here on Monday Night '' `` Heavy Action '' `` Start Me Up '' `` The Party 's Over '' Yearly results Monday night NFL
games prior to 1970 1970 -- 1989 1990 -- 2009 2010 -- present National Football League on television and radio Broadcast partners ABC ( AFL ) CBS Fox NBC
Defunct networks DuMont Sports Network Monday Night Football History Monday Night Countdown Results Games prior to 1970 1970 -- 1989 1990 -- 2009 2010 --
present Sunday Night Football ESPN NBC TNT Results ESPN NBC TNT Pregame TV programs The NFL Today CBS Fox NFL Sunday ESPN Sunday NFL Countdown Monday Night
Countdown NBC The NFL on NBC pregame show Football Night in America NFL Network List of programs NFL AM NFL Classics NFL GameDay NFL Replay NFL Total Access The
Timeline Thursday Night Football Results ( 2006 -- present ) 2007 New England Patriots -- New York Giants game Miracle in Motown NFL Films TV programs Football
Follies A Football Life Full Color Football Hard Knocks Inside the NFL NFL Dream Season NFL Films Game of the Week NFL 's Greatest Games NFL Matchup NFL Top 10
NFL Top 100 Other
</item 3>

<item 4>
Monday Night Football
The layout is standard , with teams on the background of their color , with timeout indicators also displaying in team colors . Experimentally , the score bar
`` auto - hides '' itself during plays , though it is shown permanently during the final two minutes of each half and , if necessary , overtime . This is also
the first time that ESPN has used the full 16 : 9 frame for its NFL graphics . Scoring records ( edit ) See also : List of Monday Night Football results ( 1970
-- 89 ) , List of Monday Night Football results ( 1990 -- 2009 ) , and List of Monday Night Football results ( 2010 -- present ) Most points 59 -- Philadelphia
Eagles , November 15 , 2010 vs. Washington Redskins 55 -- Indianapolis Colts , October 31 , 1988 vs. Denver Broncos 52 -- San Francisco 49ers , December 23 ,
1991 vs. Chicago Bears 52 -- San Diego Chargers , December 20 , 1982 vs. Cincinnati Bengals 51 -- New Orleans Saints , November 24 , 2008 vs.
</item 4>

<item 5>
Monday Night Football
6 2016 summary 2.2. 7 2017 summary 3 Theme music 4 Digital on - screen graphics 4.1 1997 -- 2005 4.2 2006 -- present 5 Scoring records 6 Scheduling on local
stations 7 Commentators 8 Foreign - language versions 8.1 Spanish version 8.2 Portuguese version 9 Radio broadcasts 10 Nielsen ratings 11 Additional NFL game
rights 11.1 Playoff games 11.2 Pro Bowl 12 See also 13 Footnotes 14 References 15 Further reading 16 External links Overview ( edit ) This section needs
expansion . You can help by adding to it . ( November 2010 ) See also : History of Monday Night Football After 45 seasons , there now have been a total of 730
games televised by the Monday Night Football franchise . Scheduling problems ( edit ) To avoid any scheduling unfairness where , just before the first playoff
game , a team may have five days off and others six , there is no Monday night game during the final week of the regular season .
</item 5>

<item 6>
Monday Night Football
go.com/nfl/mnf ESPN Monday Night Football ( abbreviated as MNF and also known as ESPN Monday Night Football on ABC for rare live special broadcast ) is a live
television broadcast of weekly National Football League ( NFL ) games on ESPN in the United States . From 1970 to 2005 , it aired on sister broadcast network
ABC . Monday Night Football was , along with Hallmark Hall of Fame and the Walt Disney anthology television series , one of the longest - running prime time
programs ever on commercial network television , and one of the highest - rated , particularly among male viewers . MNF is preceded on air by Monday Night
Countdown served by Applebee 's . Monday Night Football is also broadcast in Canada on TSN and RDS , and in most of Europe . On September 7 , 2013 , the NFL
announced that British Eurosport would show Monday Night Football games live in the United Kingdom for the 2013 and 2014 seasons . The telecasts are also seen
in most of Australia on ESPN Australia , in Portugal on SportTV 3 and SportTV HD and on TV 2 Sport in Denmark , and in some other regions of the world outside
the U.S.
</item 6>

<item 7>
Monday Night Football
Monday Night Football - Wikipedia Monday Night Football Jump to : navigation , search For other uses , see Monday Night Football ( disambiguation ) . `` NFL on
ESPN '' redirects here . For the program aired between 1987 and 2005 , see ESPN Sunday Night Football . An automated process has detected links on this page on
the local or global blacklist . If the links are appropriate you may request whitelisting by following these instructions ; otherwise consider removing or
replacing them with more appropriate links . ( To hide this tag , set the `` invisible '' field to `` true '' ) List of blacklisted links : ( show )
http://nfllabor.files.wordpress.com/2013/12/week-161.pdf Triggered by \ bfiles \. wordpress \. com \ b on the global blacklist
https://web.archive.org/web/20131224083130/http://nfllabor.files.wordpress.com/2013/12/week-161.pdf Triggered by \ bfiles \. wordpress \.
</item 7>

<item 8>
Monday Night Football
wordpress \. com \ b on the global blacklist ESPN Monday Night Football Genre NFL game telecasts Created by Roone Arledge Directed by Chet Forte ( 1970 -- 1986
) Larry Kamm ( 1987 ) Craig Janoff ( 1988 -- 1999 ) Drew Esocoff ( 2000 -- 2005 ) Chip Dean ( 2006 -- present ) Presented by Commentators : Sean McDonough Lisa
Salters Studio hosts : Suzy Kolber Steve Young Randy Moss Matt Hasselbeck Charles Woodson Opening theme `` Score '' by Bob 's Band ( 1970 -- 1975 ) `` All My
Rowdy Friends Are Here on Monday Night '' by Hank Williams , Jr. ( 1989 -- 2011 ; 2017 -- present ) `` Heavy Action '' by Johnny Pearson ( 1976 -- 1988 ; 2011
-- 2016 ) ( used as secondary theme from 1989 to 2011 and again from 2017 -- present ) Country of origin United States Original language ( s ) English No. of
seasons 47 ( NFL seasons ) No.
</item 8>

<item 9>
Monday Night Football
However , the forecasting abilities of the NFL 's scheduling officials have exhibited uncanny proficiency . Many MNF games occurring later in the season feature
at least one team that is either headed to the playoffs or must win the MNF game to clinch a playoff spot . The problem of having a national spotlight game
which during the season 's most critical weeks late in the year probably would not show the most important game of the week was long known by the league and
network . As a result of this , the NFL wished to move the `` Game of the Week '' idea to Sunday nights as to make flex scheduling possible . This was a move
which would ultimately mean the end of Monday Night Football on ABC ( cable games are protected from the NFL 's flexible scheduling rule adopted for the 2006 --
07 season ; the new rule applies only to CBS , Fox , and NBC 's Sunday night games ) . The league currently has no interest in expanding flexible scheduling to
include Monday Night Football , citing the logistical issues of moving games back and forth between Sundays and Mondays . Franchises with the most appearances (
edit ) The franchise with the most Monday night appearances is the Miami Dolphins .
</item 9>

<item 10>
Monday Night Football
From 2003 to 2005 , one game was played on Thursday and another Monday under the Monday Night Football banner . Starting in 2006 , when the series moved to
cable , two games are played on the opening Monday night to capitalize on fan interest during `` NFL Kickoff Weekend '' . Monday night games early in the season
are often highly anticipated since records are new , teams usually are showcasing fresh talent and potential , and storylines coming into the season are often
played out as fans try to see if these hyped teams are up to form . Since no one knows during the first month of the season if a team is indeed good , or will
rebound from a difficult start , interest is usually high for the first few weeks of the MNF season . Since the MNF schedule is set in April and can not be
changed , the league and network can not guarantee a late season matchup will have any significance or be highly anticipated .
</item 10>
=== END ITEM id=45b612f12c8daa6b ===

=== ITEM id=9793b0ab151bfc32 ===
Gold sentences and facts:
Sentence 0: Rank Team Won Lost Tied Pct . First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744 568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684 572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens 181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401 11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400 460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392 468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32 Tampa Bay Buccaneers 250 393 . 389 1976 644 NFC South
  Fact 0.0: The table columns are rank, team, won, lost, tied, pct, first NFL season, total games, and division.
  Fact 0.1: The Dallas Cowboys won 493 games, lost 367, tied 6, had a pct of .573, had a first NFL season of 1960, played 866 total games, and are in the NFC East.
  Fact 0.2: The Green Bay Packers won 730 games, lost 553, tied 37, had a pct of .567, had a first NFL season of 1921, played 1,320 total games, and are in the NFC North.
  Fact 0.3: The Chicago Bears won 744 games, lost 568, tied 42, had a pct of .565, had a first NFL season of 1920, played 1,354 total games, and are in the NFC North.
  Fact 0.4: The Miami Dolphins won 439 games, lost 341, had a pct of .563, had a first NFL season of 1966, played 784 total games, and are in the AFC East.
  Fact 0.5: Rank 5 is the New England Patriots, who won 476 games, lost 383, tied 9, had a pct of .554, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.6: Rank 6 is the New York Giants, who won 684 games, lost 572, tied 33, had a pct of .543, had a first NFL season of 1925, played 1,289 total games, and are in the NFC East.
  Fact 0.7: Rank 7 is the Denver Broncos, who won 465 games, lost 393, tied 10, had a pct of .541, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.8: Rank 8 is the Minnesota Vikings, who won 457 games, lost 387, tied 10, had a pct of .541, had a first NFL season of 1961, played 854 total games, and are in the NFC North.
  Fact 0.9: Rank 9 is the Baltimore Ravens, who won 181 games, lost 154, had a pct of .540, played 336 total games, and are in the AFC North.
  Fact 0.10: Rank 10 is the San Francisco 49ers, who won 522 games, lost 450, tied 14, had a pct of .537, had a first NFL season of 1950, played 986 total games, and are in the NFC West.
  Fact 0.11: Rank 11 is the Indianapolis Colts, who won 502 games, lost 441, tied 7, had a pct of .532, had a first NFL season of 1953, played 950 total games, and are in the AFC South.
  Fact 0.12: Rank 12 is the Oakland Raiders, who won 456 games, lost 401, tied 11, had a pct of .532, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.13: Rank 13 is the Pittsburgh Steelers, who won 601 games, lost 535, tied 20, had a pct of .529, had a first NFL season of 1933, played 1,156 total games, and are in the AFC North.
  Fact 0.14: Rank 14 is the Kansas City Chiefs, who won 447 games, lost 409, tied 12, had a pct of .522, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.15: Rank 15 is the Washington Redskins, who won 586 games, lost 572, tied 28, had a pct of .506, had a first NFL season of 1932, played 1,186 total games, and are in the NFC East.
  Fact 0.16: Rank 16 is the Seattle Seahawks, who won 325 games, lost 318, had a pct of .505, had a first NFL season of 1976, played 644 total games, and are in the NFC West.
  Fact 0.17: Rank 17 is the Cleveland Browns, who won 462 games, lost 466, tied 10, had a pct of .498, had a first NFL season of 1950, played 938 total games, and are in the AFC North.
  Fact 0.18: Rank 18 is the Los Angeles Chargers, who won 426 games, lost 431, tied 11, had a pct of .497, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.19: Rank 19 is the Los Angeles Rams, who won 544 games, lost 554, tied 21, had a pct of .496, had a first NFL season of 1937, played 1,119 total games, and are in the NFC West.
  Fact 0.20: Rank 20 is the Carolina Panthers, who won 172 games, lost 179, had a pct of .490, had a first NFL season of 1995, played 352 total games, and are in the NFC South.
  Fact 0.21: Rank 21 is the Philadelphia Eagles, who won 555 games, lost 591, tied 26, had a pct of .485, had a first NFL season of 1933, played 1,172 total games, and are in the NFC East.
  Fact 0.22: Rank 22 is the Tennessee Titans, who won 413 games, lost 449, tied 6, had a pct of .479, had a first NFL season of 1960, played 868 total games, and are in the AFC South.
  Fact 0.23: Rank 23 is the Buffalo Bills, who won 400 games, lost 460, tied 8, had a pct of .465, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.24: Rank 24 is the Detroit Lions, who won 544 games, lost 641, tied 32, had a pct of .460, had a first NFL season of 1930, played 1,217 total games, and are in the NFC North.
  Fact 0.25: Rank 25 is the Cincinnati Bengals, who won 344 games, lost 408, had a pct of .458, had a first NFL season of 1968, played 756 total games, and are in the AFC North.
  Fact 0.26: Rank 26 is the New York Jets, who won 392 games, lost 468, tied 8, had a pct of .456, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.27: Rank 27 is the New Orleans Saints, who won 338 games, lost 427, tied 5, had a pct of .442, had a first NFL season of 1967, played 770 total games, and are in the NFC South.
  Fact 0.28: Rank 28 is the Houston Texans, who won 106 games, lost 134, tied 0, had a pct of .442, had a first NFL season of 2002, played 240 total games, and are in the AFC South.
  Fact 0.29: Rank 29 is the Jacksonville Jaguars, who won 155 games, lost 197, tied 0, had a pct of .440, had a first NFL season of 1995, played 352 total games, and are in the AFC South.
  Fact 0.30: Rank 30 is the Atlanta Falcons, who won 341 games, lost 437, tied 6, had a pct of .439, had a first NFL season of 1966, played 784 total games, and are in the NFC South.
  Fact 0.31: Rank 31 is the Arizona Cardinals, who won 542 games, lost 732, tied 40, had a pct of .428, had a first NFL season of 1920, played 1,314 total games, and are in the NFC West.
  Fact 0.32: Rank 32 is the Tampa Bay Buccaneers, who won 250 games, lost 393, had a pct of .389, had a first NFL season of 1976, played 644 total games, and are in the NFC South.

Retrieved items:
<item 1>
NFL win–loss records
480 AFC East 18 Philadelphia Eagles 40 19 21 . 475 2013 NFC East 19 Jacksonville Jaguars 11 5 6 . 455 2007 AFC South 20 Los Angeles Rams 43 19 24 . 442 NFC West
21 ( tie ) Arizona Cardinals 16 7 9 . 438 2015 NFC West 21 ( tie ) New Orleans Saints 16 7 9 . 438 2013 NFC South 23 Houston Texans 7 . 429 2016 AFC South 24
Tennessee Titans 33 14 19 . 424 2008 AFC South 25 Atlanta Falcons 22 9 13 . 409 2016 NFC South 26 Minnesota Vikings 47 19 28 . 404 2015 NFC North 27 Tampa Bay
Buccaneers 15 6 9 . 400 2007 NFC South 28 Los Angeles Chargers 28 11 17 . 393 2013 AFC West 29 Cleveland Browns 31 11 20 . 355 2002 AFC North 30 Detroit Lions
20 7 13 .
</item 1>

<item 2>
NFL win–loss records
355 2002 AFC North 30 Detroit Lions 20 7 13 . 350 2016 NFC North 31 Kansas City Chiefs 26 9 17 . 346 2016 AFC West 32 Cincinnati Bengals 19 5 14 . 263 2015 AFC
North Notes ( edit ) Jump up ^ For example , the Browns ' regular season win -- loss record does not include a 47 -- 4 -- 3 record compiled in the All - America
Football Conference from 1946 to 1949 , nor does their playoff record include the 5 -- 0 record compiled during the same period . Likewise , the 49ers ' AAFC
regular season win -- loss record of 38 -- 14 -- 2 , nor their 1 - 1 AAFC playoff record are officially counted Jump up ^ The Patriots played as the `` Boston
Patriots '' from 1960 -- 1970 , before adopting the `` New England Patriots '' name , which they have played under since the 1971 season . The team was also
briefly known as the `` Bay State Patriots '' prior to the 1971 season .
</item 2>

<item 3>
List of National Football League records (individual)
Peyton Manning , Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders , 2014 Most losses in a season , by a starting quarterback : 15 , Archie Manning , New
Orleans Saints , 1980 ; Jeff George , Indianapolis Colts , 1991 ; Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses , by a rookie starting
quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most losses in a season , by a rookie starting quarterback : 15 , Deshone Kizer , Cleveland Browns ,
2017 Most consecutive losses in a season , by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses to start a season ,
by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses by a starting quarterback : 21 , Dan Pastorini , Houston Oilers
, 1972 -- 1974 Most consecutive losses by a starting quarterback , regular season , to start a career : 15 , Deshone Kizer , Cleveland Browns , 2017 Most
consecutive road losses by a starting quarterback , regular season , to start a career : 13 ,
</item 3>

<item 4>
The article ranks all 32 active NFL teams by regular-season records through 2016 and playoff records through Super Bowl LI. Ties count as half a win and half a
loss; AFL results are included, but AAFC results are excluded. Dallas led regular-season winning percentage (.573), followed by Green Bay and Chicago. Baltimore
led playoff percentage (.652), while Cincinnati ranked last (.263). Tampa Bay had the lowest regular-season percentage (.389).
</item 4>

<item 5>
NFL win–loss records
NFL win -- loss records - wikipedia NFL win -- loss records Jump to : navigation , search It has been suggested that NFL standings since AFL -- NFL merger be
merged into this article . ( Discuss ) Proposed since October 2017 . The following is a list of win -- loss records for each of the 32 active National Football
League ( NFL ) teams . Contents ( hide ) 1 Regular season 2 Playoffs 3 Notes 4 References Regular season ( edit ) George Halas ( right ) of the Chicago Bears ,
who have recorded the most regular season wins . The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their regular
season win -- loss record percentage , accurate as of the end of the 2016 NFL regular season . For teams that previously played in the All - America Football
Conference ( AAFC ) , their AAFC win - loss records are not included as the NFL does not officially count AAFC statistics despite the 1950 NFL -- AAFC merger .
However , for teams that played in the American Football League ( AFL ) , their AFL win - loss records are included and officially counted by the NFL .
</item 5>

<item 6>
NFL win–loss records
600 2013 NFC West 6 Oakland Raiders 44 25 19 . 568 2016 AFC West 7 Carolina Panthers 16 9 7 . 563 2015 NFC South 8 Dallas Cowboys 61 34 27 . 557 2016 NFC East 9
( tie ) Washington Redskins 42 23 19 . 548 2015 NFC East 9 ( tie ) Denver Broncos 42 23 19 . 548 2015 AFC West 11 Seattle Seahawks 31 16 15 . 516 2016 NFC West
12 New York Giants 49 24 25 . 490 2016 NFC East 13 Indianapolis Colts 45 22 23 . 489 2014 AFC South 14 Miami Dolphins 41 20 21 . 488 2016 AFC East 15 Chicago
Bears 35 17 18 . 486 NFC North 16 Buffalo Bills 29 14 15 . 483 1999 AFC East 17 New York Jets 25 12 13 . 480 AFC East 18 Philadelphia Eagles 40 19 21 .
</item 6>

<item 7>
NFL win–loss records
389 1976 644 NFC South Playoffs ( edit ) Members of the Baltimore Ravens franchise , the team with the highest playoff win -- loss percentage , during the
Lombardi Trophy presentation following their Super Bowl XLVII victory Marvin Lewis , head coach of the Cincinnati Bengals , who have registered the lowest
playoff win - loss percentage of all time The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their playoff win - loss
record percentage , accurate as of the end of Super Bowl LI . Best win -- loss record in division Team active in playoffs Rank Team Games Won Lost Pct . Last
playoff appearance Division Baltimore Ravens 23 15 8 . 652 2014 AFC North New England Patriots 51 32 19 . 627 2016 AFC East Green Bay Packers 56 34 22 . 607
2016 NFC North 4 ( tie ) Pittsburgh Steelers 60 36 24 . 600 2016 AFC North 4 ( tie ) San Francisco 49ers 50 30 20 . 600 2013 NFC West 6 Oakland Raiders 44 25 19
.
</item 7>

<item 8>
NFL win–loss records
Ties are registered as a half win and a half loss when calculating the win - loss record . Best win -- loss record in division Rank Team Won Lost Tied Pct .
First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744
568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684
572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens
181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 .
</item 8>

<item 9>
NFL win–loss records
490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400
460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392
468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville
Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32
Tampa Bay Buccaneers 250 393 .
</item 9>

<item 10>
NFL win–loss records
540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401
11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington
Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los
Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC
South 21 Philadelphia Eagles 555 591 26 .
</item 10>
=== END ITEM id=9793b0ab151bfc32 ===

=== ITEM id=524c704db8397ef7 ===
Gold sentences and facts:
Sentence 0: Dragon Ball Z ( ドラゴンボール ゼット , Doragon Bōru Zetto , commonly abbreviated as DBZ ) is the long - running anime sequel to the Dragon Ball TV series , adapted from the final twenty - six volumes of the Dragon Ball manga written by Akira Toriyama . The manga portion of the series debuted in Weekly Shōnen Jump in October 4 , 1988 and lasted until 1995 ; the anime adaptation premiered in Japan on Fuji Television on April 26 , 1989 , taking over its predecessor 's time slot , and ran until its end on January 31 , 1996 , lasting 291 episodes in Japan , and 276 episodes in the United States originally , although all 291 episodes were later broadcast when content from the first 67 episodes was restored .
  Fact 0.0: Dragon Ball Z (ドラゴンボール ゼット, Doragon Bōru Zetto, commonly abbreviated as DBZ) is the long-running anime sequel to the Dragon Ball TV series.
  Fact 0.1: Dragon Ball Z is adapted from the final twenty-six volumes of the Dragon Ball manga written by Akira Toriyama.
  Fact 0.2: The Dragon Ball Z manga portion debuted in Weekly Shōnen Jump in October 4, 1988.
  Fact 0.3: The Dragon Ball Z manga portion lasted until 1995.
  Fact 0.4: The Dragon Ball Z anime adaptation premiered in Japan on Fuji Television on April 26, 1989.
  Fact 0.5: The Dragon Ball Z anime adaptation took over its predecessor's time slot.
  Fact 0.6: The Dragon Ball Z anime adaptation ended on January 31, 1996.
  Fact 0.7: The Dragon Ball Z anime adaptation lasted 291 episodes in Japan.
  Fact 0.8: The Dragon Ball Z anime adaptation originally lasted 276 episodes in the United States.
  Fact 0.9: All 291 Dragon Ball Z episodes were later broadcast, after content from the first 67 episodes was restored.

Retrieved items:
<item 1>
List of Dragon Ball Z episodes
( Saiyan Sized Secret ) `` Ima da Gokū ! Subete o Kaketa Saigo no Ōwaza '' ( いまだ 悟空 ! すべて を 賭け た 最後 の 大 技 ) December 20 , 1989 May 3 , 1997 August 4 , 2005 (
re-dub ) 32 24 `` Battle Power Times Ten ! ! Vegeta 's Great Metamorphosis '' / `` Vegeta ... Saiyan Style ! '' ( Spirit Bomb Away ! ) `` Sentōryoku Jūbai ! !
Bejīta Daihenshin '' ( 戦闘 力 10 倍 ! ! ベジータ 大 変身 ) January 17 , 1990 May 10 , 1997 August 8 , 2005 ( re-dub ) 33 25 `` Do n't Die , Father ! ! This is the Depth
of Gohan 's Power '' / `` Stop Vegeta Now ! ''
</item 1>

<item 2>
List of Dragon Ball Z episodes
! You 're Finished Now , Majin Boo '' / `` Time Struggle '' `` Tokkun Seikō ! ! Kore de Owari da Majin Bū '' ( 特訓 成功 ! ! これ で 終り だ 魔 人 ブウ ) February 22 , 1995
October 17 , 2002 258 243 `` I 'm Going Like I Mean It ! ! A Wide - Open Super Gotenks '' / `` Super Moves of Gotenks '' `` Honki de Iku ze ! ! Sūpā Gotenkusu
Zenkai '' ( 本気 で 行く ぜ ! ! 超 ゴテンクス 全開 ) March 1 , 1995 October 21 , 2002 259 244 `` I 've Done It ! ! Boo Successfully Eliminated With Ghosts ! ? '' / `` Trapped
in Forever '' `` Yatta ze ! ! Obake de Seikō Bū Taiji ! ? ''
</item 2>

<item 3>
List of Dragon Ball Z episodes
! ブッ 飛べ ヤコン ) April 13 , 1994 October 18 , 2001 224 209 `` A Great Miscalculation ! ! Satan vs. Three Super-Warriors ! ? '' / `` Battle Supreme '' `` Daigosan !
! Satan tai Sannin no Chōsenshi ! ? '' ( 大 誤算 ! ! サタン VS3 人 の 超 戦士 ! ? ) April 20 , 1994 October 22 , 2001 225 210 `` Such Strong Kids ! ! No. 18 's Close Fight
! ? '' / `` Eighteen Unmasks '' `` Tsuyoi ze Chibikko ! ! Jūhachigō Daikusen ! ? '' ( 強い ぜ チビッコ ! ! 18 号 大 苦戦 ! ? )
</item 3>

<item 4>
List of Dragon Ball Z episodes
! We 'll Make a Huge Genki Dama '' / `` Call to Action '' `` Saigo no Kibō ! ! Tsukuru ze Dekkai Genkidama '' ( 最後 の 希望 ! ! 作る ぜ でっかい 元気 玉 ) November 22 , 1995
March 26 , 2003 285 270 `` Ultra-Impressive ! ! The Genki Dama From Everyone is Finished '' / `` People of Earth Unite '' `` Chōkangeki ! ! Dekita ze Minna no
Genkidama '' ( 超 感激 ! ! でき た ぜ みんなの 元気 玉 ) November 29 , 1995 March 27 , 2003 286 271 `` Son Goku is Strongest After All ! ! Majin Boo is Eliminated '' / ``
Spirit Bomb Triumphant '' `` Yappari Saikyō Son Gokū ! !
</item 4>

<item 5>
List of Dragon Ball Z episodes
The series was broadcast in eight separate near - continuous blocks ( 1 - 35 , 36 - 67 , 68 - 107 , 108 - 116 , 117 - 194 , 195 - 237 , 238 - 276 , and 277 -
291 ) , with breaks varying between four months to over a year between each block . In only one instance ( between episodes 194 and 195 ) was there actually
parity between the DVD release and the actual broadcast sequence in terms of the end of one `` season '' and the beginning of the next .
</item 5>

<item 6>
List of Dragon Ball Z episodes
The manga portion of the series debuted in Weekly Shōnen Jump in October 4 , 1988 and lasted until 1995 ; the anime adaptation premiered in Japan on Fuji
Television on April 26 , 1989 , taking over its predecessor 's time slot , and ran until its end on January 31 , 1996 , lasting 291 episodes in Japan , and 276
episodes in the United States originally , although all 291 episodes were later broadcast when content from the first 67 episodes was restored . Dragon Ball Z
uses four pieces of theme music in the Japanese version . From episodes 1 - 199 , the opening theme is `` Cha - La Head - Cha - La '' by Hironobu Kageyama , and
the closing theme is `` Detekoi Tobikiri Zenkai Pawā ! '' by MANNA . From episodes 200 - 291 , the opening and closing themes are `` We Gotta Power '' and ``
Boku - Tachi wa Tenshi Datta , '' both by Hironobu Kagayama .
</item 6>

<item 7>
List of Dragon Ball Z episodes
? '' ( 炎 の 決勝 ! ! 悟空 か パイクーハン か ! ? ) August 25 , 1993 September 13 , 2001 199 184 `` Do n't Let Victory Get Away ! ! Finish With an Ultra-Fast Kamehame - Ha ''
/ `` Goku vs. Pikkon '' `` Nigasu na Shōri ! ! Kimero Chōhaya Kamehameha '' ( 逃す な 勝利 ! ! 決めろ 超速 かめ はめ 波 ) September 1 , 1993 September 14 , 2001 200 185 ``
Seven Years Since Then ! Starting Today , I 'm a High School Student '' / `` Gohan Goes to High School '' `` Are kara Shichinen ! Kyō kara Boku wa Kōkōsei '' (
あれ から 7 年 !
</item 7>

<item 8>
List of Dragon Ball Z episodes
Jump up ^ `` Dragon Ball Z Season 7 '' . Amazon.co.uk . Retrieved June 27 , 2014 . Jump up ^ `` Dragon Ball Z Season 8 '' . Amazon.co.uk . Retrieved June 27 ,
2014 . Jump up ^ `` Dragon Ball Z Season 9 '' . Amazon.co.uk . Retrieved June 27 , 2014 . External links ( edit ) Wikimedia Commons has media related to Dragon
Ball . Official Funimation Entertainment Dragon Ball Z episode list Dragon Ball Z ( hide ) Dragon Ball by Akira Toriyama Media Manga chapters Dragon Ball Z
Dragon Ball episodes Z episodes season 1 5 6 7 8 9 GT episodes Z Kai episodes Super episodes Films and TV specials Video games Soundtracks Characters Goku Bulma
Master Roshi Yamcha Krillin Tien Shinhan Piccolo Gohan Vegeta Frieza Trunks Android 17 Android 18 Cell Majin Buu Beerus Zamasu Related Dr. Slump Arale Norimaki
List of Dr.
</item 8>

<item 9>
List of Dragon Ball Z episodes
! お め え だけ は オラ が 倒す ) April 17 , 1991 October 6 , 1999 88 73 `` The Two Superpowers Collide ! A Fistfight Where Both Turn Serious ! ! ! '' / `` Clash of the
Super Powers '' `` Gekitotsu no Ni Dai Chō Pawā ! Honki Dōshi no Nikudan Sen ! ! ! '' ( 激突 の 2 大 超 パワー ! 本気 同士 の 肉弾 戦 ! ! ! ) April 24 , 1991 October 7 , 1999
89 74 `` Freeza 's Terrible Declaration ! I Will Defeat You Without Using My Hands '' / `` Frieza 's Boast '' `` Furīza Kyōfu no Sengen ! Te o Tsukawazu Omae o
Taosu '' ( フリーザ 恐怖 の 宣言 !
</item 9>

<item 10>
List of Dragon Ball Z episodes
! Yabō wa Yurusanai '' ( 待っ てろ バビディ ! ! 野望 は 許さ ない ) June 22 , 1994 October 31 , 2001 231 216 `` The Seal is Broken ! Out Comes the Vicious Majin Boo ! ! '' /
`` Magic Ball of Buu '' `` Toketa Fūin ! Deruzo Kyōaku Majin Bū ! ! '' ( 解け た 封印 ! 出る ぞ 凶悪 魔 人 ブウ ! ! ) June 29 , 1994 November 1 , 2001 232 217 `` I Wo n't
Allow the Revival ! ! A Kamehame - Ha of Resistance '' / `` Buu is Hatched ! '' `` Fukkatsu Sasenai ! ! Teikō no Kamehameha '' ( 復活 させない ! !
</item 10>
=== END ITEM id=524c704db8397ef7 ===

=== ITEM id=723a089c74680b44 ===
Gold sentences and facts:
Sentence 0: All black military pilots who trained in the United States trained at Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic .
  Fact 0.0: All black military pilots who trained in the United States trained at Moton Field.
  Fact 0.1: All black military pilots who trained in the United States trained at the Tuskegee Army Air Field.
  Fact 0.2: All black military pilots who trained in the United States were educated at Tuskegee University.
  Fact 0.3: Tuskegee University is located near Tuskegee, Alabama.
  Fact 0.4: The group included five Haitians from the Haitian Air Force.
  Fact 0.5: The group included one pilot from Trinidad.
  Fact 0.6: The group included a Hispanic or Latino airman born in the Dominican Republic.

Retrieved items:
<item 1>
The Tuskegee Airmen—Black pilots and support crews—trained at Tuskegee and bases nationwide after Congress funded Black flight training in 1939. The 99th
Squadron deployed to North Africa in 1943, later fighting in Sicily and Italy; the 332nd became the first Black fighter group. The 477th trained B-25 crews but
never saw combat. Despite segregation and discrimination, airmen challenged whites-only policies, notably at Freeman Field in 194
</item 1>

<item 2>
Tuskegee Airmen
Finally , on 3 April 1939 , Appropriations Bill Public Law 18 was passed by Congress containing an amendment by Senator Harry H. Schwartz , designating funds
for training African - American pilots . The War Department managed to put the money into funds of civilian flight schools willing to train black Americans .
War Department tradition and policy mandated the segregation of African - Americans into separate military units staffed by white officers , as had been done
previously with the 9th Cavalry , 10th Cavalry , 24th Infantry Regiment and 25th Infantry Regiment . When the appropriation of funds for aviation training
created opportunities for pilot cadets , their numbers diminished the rosters of these older units . In 1941 , the War Department and the Army Air Corps , under
pressure -- three months before its transformation into the USAAF -- constituted the first all - black flying unit , the 99th Pursuit Squadron . Due to the
restrictive nature of selection policies , the situation did not seem promising for African - Americans since , in 1940 , the U.S. Census Bureau reported there
were only 124 African - American pilots in the nation .
</item 2>

<item 3>
The Tuskegee Airmen became the first African Americans to serve as U.S. military pilots. Their path followed the rejection of Black men seeking aviation roles
in 1917; Eugene Bullard instead flew for France during World War I. Decades of advocacy by NAACP leader Walter White, labor leader A. Philip Randolph, and Judge
William H. Hastie helped advance the effort. Images show a Red Tail Project P-51C, a Stearman named *Spirit of Tuske
</item 3>

<item 4>
In 1939, Congress funded flight training for African Americans, with money directed to civilian schools. Despite military segregation and only 124 Black pilots
reported in 1940, applications exceeded expectations, including from Civilian Pilot Training Program participants; Tuskegee University joined the program in
1939. In 1941, the Army Air Corps established the first all-Black flying unit, the 99th Pursuit Squadron. Army aviation-cadet training included psychological
testing and selection for pilots,
</item 4>

<item 5>
Tuskegee Airmen
They formed the 332nd Fighter Group and the 477th Bombardment Group of the United States Army Air Forces . The name also applies to the navigators , bombardiers
, mechanics , instructors , crew chiefs , nurses , cooks and other support personnel . All black military pilots who trained in the United States trained at
Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from
the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic . Although the 477th
Bombardment Group trained with North American B - 25 Mitchell bombers , they never served in combat . The 99th Pursuit Squadron ( later , 99th Fighter Squadron
) was the first black flying squadron , and the first to deploy overseas ( to North Africa in April 1943 , and later to Sicily and Italy ) . The 332nd Fighter
Group , which originally included the 100th , 301st , and 302nd Fighter Squadrons , was the first black flying group .
</item 5>

<item 6>
The Tuskegee Airmen were the first African-American military aviators in the U.S. Armed Forces. They served during World War II, when the military and much of
the country remained racially segregated under Jim Crow laws, and faced discrimination inside and outside the Army. Their P-51 Mustangs were marked with red
tails and propeller spinners, red nose bands, and yellow wing bands. The article covers their origins, training, combat service, accomplishments, postwar
legacy, and cultural
</item 6>

<item 7>
The Tuskegee Airmen included Black pilots and support personnel who trained at Moton Field and Tuskegee Army Air Field and studied at Tuskegee University. The
99th was the first Black flying squadron and deployed to North Africa in April 1943, later serving in Sicily and Italy. The 332nd was the first Black flying
group. The 477th trained on B-25 bombers but never saw combat. The airmen also included Haitians, a Trinidad
</item 7>

<item 8>
Tuskegee Airmen
Some ground crews trained at Mather before rotating to Inglewood . Gunners learned to shoot at Eglin Field , Florida . Bombers - navigators learned their trades
at Hondo Army Air Field and Midland Air Field , Texas or at Roswell , New Mexico . Training of the new African - American crewmen also took place at Sioux Falls
, South Dakota , Lincoln , Nebraska , and Scott Field , Belleville , Illinois . Once trained , the air and ground crews would be spliced into a working unit at
Selfridge . Command difficulties ( edit ) The new group 's first commanding officer was Colonel Robert Selway , who had also commanded the 332nd Fighter Group
before it deployed for combat overseas . Like his ranking officer , Major General Frank O'Driscoll Hunter from Georgia , Selway was a racial segregationist .
Hunter was blunt about it , saying such things as , `` ... racial friction will occur if colored and white pilots are trained together . '' He backed Selway 's
violations of Army Regulation 210 - 10 , which forbade segregation of air base facilities .
</item 8>

<item 9>
Tuskegee Airmen
The white population of Freeman Field was 250 officers and 600 enlisted men . Superimposed on it were 400 African - American officers and 2,500 enlisted men of
the 477th and its associated units . Freeman Field had a firing range , usable runways , and other amenities useful for training . African - American airmen
would work in proximity with white ones ; both would live in a public housing project adjacent to the base . Colonel Selway turned the non-commissioned officers
out of their club and turned it into a second officers club . He then classified all white personnel as cadre , and all African - Americans as trainees . One
officers club became the cadre 's club . The old Non-Commissioned Officers Club , promptly sarcastically dubbed `` Uncle Tom 's Cabin '' , became the trainee 's
officers club . At least four of the trainees had flown combat in Europe as fighter pilots , and had about four years in service . Four others had completed
training as pilots , bombardiers and navigators , and may have been the only triply qualified officers in the entire Air Corps .
</item 9>

<item 10>
General Henry “Hap” Arnold opposed assigning African American pilots to existing Air Corps units, arguing that Black officers serving over white enlisted men
would create an “impossible social situation.” The 99th Fighter Squadron was deemed combat-ready in April 1943 and left Tuskegee on April 2 for North Africa,
joining the 33rd Fighter Group. With little guidance from combat-experienced pilots, it began operations against Pantelleria on June 2, helping clear
Mediterranean sea lanes for the
</item 10>
=== END ITEM id=723a089c74680b44 ===

=== ITEM id=82452832b3f6efd5 ===
Gold sentences and facts:
Sentence 0: Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 % WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X 10.0 -- 10.1. 5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS / 2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS. TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts
  Fact 0.0: The hosts file on Unix, Unix-like, and POSIX systems is at /etc/hosts
  Fact 0.1: On Microsoft Windows 3.1 the hosts file is at %WinDir%\HOSTS
  Fact 0.2: On Windows 95, 98, and ME the hosts file is at %WinDir%\hosts
  Fact 0.3: On Windows NT, 2000, XP, 2003, Vista, 2008, 7, 2012, 8, and 10 the hosts file is at %SystemRoot%\System32\drivers\etc\hosts
  Fact 0.4: On Windows Mobile and Windows Phone the hosts entry is a registry key under HKEY_LOCAL_MACHINE\Comm\Tcpip\Hosts
  Fact 0.5: On Apple Macintosh 9 and earlier the hosts file is in the Preferences or System folder
  Fact 0.6: On Mac OS X 10.0 to 10.1.5 the hosts file was added through NetInfo or niload
  Fact 0.7: On Mac OS X 10.2 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.8: On Novell NetWare the hosts file is at SYS:etc\hosts
  Fact 0.9: On OS/2 and eComStation the hosts file is at the bootdrive path \mptn\etc\
  Fact 0.10: On Symbian OS 6.1 to 9.0 the hosts file is at C:\system\data\hosts
  Fact 0.11: On Symbian OS 9.1 and later the hosts file is at C:\private\10000882\hosts
  Fact 0.12: On MorphOS (NetStack) the hosts file is at ENVARC:sys/net/hosts
  Fact 0.13: On AmigaOS before 4 the hosts file is at AmiTCP:db/hosts or DEVS:Internet/hosts
  Fact 0.14: On AROS the hosts file is at ENVARC:AROSTCP/db/hosts
  Fact 0.15: On Android the hosts file is /etc/hosts, a symbolic link to /system/etc/hosts
  Fact 0.16: On iOS 2.0 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.17: On TOPS-20 the hosts file is <SYSTEM>HOSTS.TXT
  Fact 0.18: On Plan 9 the hosts file is /lib/ndb/hosts
  Fact 0.19: On BeOS the hosts file is /boot/beos/etc/hosts
  Fact 0.20: On Haiku the hosts file is /boot/common/settings/network/hosts
  Fact 0.21: On OpenVMS with UCX the hosts file is UCX$HOST
  Fact 0.22: On OpenVMS with TCPware the hosts file is TCPIP$HOST
  Fact 0.23: On RISC OS 3.7 and 5 the hosts file is at !Boot.Resources.!Internet.files.Hosts, loaded in the later boot sequence
  Fact 0.24: On RISC OS the disabled hosts file is at !Boot.Choices.Hardware.Disabled.Internet.Files.Hosts

Retrieved items:
<item 1>
hosts (file)
5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS /
2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts
MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a
symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS.
</item 1>

<item 2>
hosts (file)
History of BIND . Retrieved 2017 - 07 - 01 . Jump up ^ `` Cisco Networking Academy Program : First - Year Companion Guide '' , Cisco Systems , Inc. , 2002 ( 2nd
Edition ) , page 676 , ISBN 1 - 58713 - 025 - 4 Jump up ^ `` Linux Network Administrators Guide : Writing hosts and networks files '' . Retrieved May 16 , 2010
. Jump up ^ `` Hosts File '' . Retrieved August 10 , 2011 . Jump up ^ `` Microsoft KB Q314053 : TCP / IP and NBT configuration parameters for Windows XP '' .
Retrieved August 28 , 2010 . Jump up ^ `` Microsoft KB 972034 Revision 2.0 : default hosts files '' . Retrieved August 28 , 2010 . ^ Jump up to : `` Mac OS X :
How to Add Hosts to Local Hosts File '' . Retrieved August 28 , 2010 . ^ Jump up to : `` The Haiku / BeOS Tip Server '' . Retrieved November 30 , 2012 .
</item 2>

<item 3>
hosts (file)
wikipedia.org/w/index.php?title=Hosts_(file)&oldid=833616207 '' Categories : Domain name system Configuration files Hidden categories : Articles needing
additional references from April 2017 All articles needing additional references Pages using RFC magic links Talk Contents About Wikipedia Català Deutsch
Ελληνικά Español فارسی Français 한국어 עברית Lumbaart Magyar 日本 語 Polski Português Русский Svenska தமிழ் Türkçe Українська 吴语 中文 12 more Edit links This page was
last edited on 1 April 2018 , at 16 : 33 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using
this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 3>

<item 4>
hosts (file)
Network nodes typically had one address and could have many names . As local area TCP / IP computer networks gained popularity , however , the maintenance of
hosts files became a larger burden on system administrators as networks and network nodes were being added to the system with increasing frequency .
Standardization efforts , such as the format specification of the file HOSTS. TXT in RFC 952 , and distribution protocols , e.g. , the hostname server described
in RFC 953 , helped with these problems , but the centralized and monolithic nature of hosts files eventually necessitated the creation of the distributed
Domain Name System ( DNS ) . On some old systems a file named networks is present that has similar to hosts file functions containing names of networks .
Extended applications ( edit ) In its function of resolving host names , the hosts file may be used to define any hostname or domain name for use in the local
system . Redirecting local domains Some web service and intranet developers and administrators define locally defined domains in a LAN for various purposes ,
such as accessing the company 's internal resources or to test local websites in development .
</item 4>

<item 5>
hosts (file)
TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST
RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts History ( edit ) The
ARPANET , the predecessor of the Internet , had no distributed host name database . Each network node maintained its own map of the network nodes as needed and
assigned them names that were memorable to the users of the system . There was no method for ensuring that all references to a given node in a network were
using the same name , nor was there a way to read the hosts file of another computer to automatically obtain a copy . The small size of the ARPANET kept the
administrative overhead small to maintain an accurate hosts file . Network nodes typically had one address and could have many names .
</item 5>

<item 6>
hosts (file)
Comment lines may be included ; they are indicated by an octothorpe ( # ) in the first position of such lines . Entirely blank lines in the file are ignored .
For example , a typical hosts file may contain the following : 127.0. 0.1 localhost loopback : : 1 localhost This example only contains entries for the loopback
addresses of the system and their host names , a typical default content of the hosts file . The example illustrates that an IP address may have multiple host
names ( localhost and loopback ) , and that a host name may be mapped to both IPv4 and IPv6 IP addresses , as shown on the first and second lines respectively .
Location in the file system ( edit ) The location of the hosts file in the file system hierarchy varies by operating system . It is usually named hosts ,
without an extension .
</item 6>

<item 7>
hosts (file)
It is a common part of an operating system 's Internet Protocol ( IP ) implementation , and serves the function of translating human - friendly hostnames into
numeric protocol addresses , called IP addresses , that identify and locate a host in an IP network . In some operating systems , the contents of the hosts file
is used preferentially to other name resolution methods , such as the Domain Name System ( DNS ) , but many systems implement name service switches , e.g. ,
nsswitch. conf for Linux and Unix , to provide customization . Unlike remote DNS resolvers , the hosts file is under the direct control of the local computer 's
administrator . File content ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to
reliable sources . Unsourced material may be challenged and removed . ( April 2017 ) ( Learn how and when to remove this template message ) The hosts file
contains lines of text consisting of an IP address in the first text field followed by one or more host names . Each field is separated by white space -- tabs
are often preferred for historical reasons , but spaces are also used .
</item 7>

<item 8>
hosts (file)
Hosts ( file ) - wikipedia Hosts ( file ) Jump to : navigation , search For other uses , see Host ( disambiguation ) . The computer file hosts is an operating
system file that maps hostnames to IP addresses . It is a plain text file . Originally a file named HOSTS. TXT was manually maintained and made available via
file sharing by Stanford Research Institute for the ARPANET membership , containing the hostnames and address of hosts as contributed for inclusion by member
organizations . The Domain Name System , first described in 1983 and implemented in 1984 , automated the publication process and provided instantaneous and
dynamic hostname resolution in the rapidly growing network . In modern operating systems , the hosts file remains an alternative name resolution mechanism ,
configurable often as part of facilities such as the Name Service Switch as either the primary method or as a fallback method . Contents ( hide ) 1 Purpose 2
File content 3 Location in the file system 4 History 5 Extended applications 6 Security issues 7 See also 8 References 9 External links Purpose ( edit ) The
hosts file is one of several system facilities that assists in addressing network nodes in a computer network .
</item 8>

<item 9>
hosts (file)
Retrieved November 30 , 2012 . Jump up ^ RISC OS 6.14 Jump up ^ `` Remove Trojan. Qhosts -- Symantec '' . Retrieved May 16 , 2010 . External links ( edit )
Windows TCP / IP in Windows 2000 Professional Microsoft TCP / IP Host Name Resolution Order Blocking Unwanted Connections with a Hosts File ( Creative Commons
License ) Solution if Windows stops you from modifying the Hosts file or the Lmhosts file in Windows Vista / 7 / 8 Beginners Guide on Editing Hosts File List of
Microsoft related domains that Windows wo n't redirect Mac How to block unwanted domains with a HOSTS file on Mac OS X 10.2 or later `` Gas Mask '' - Hosts file
manager for Mac OS X Linux Block unwanted advertisements with / etc / hosts file on Linux Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
hosts (file)
It is usually named hosts , without an extension . Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 %
WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts
Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X
10.0 -- 10.1.
</item 10>
=== END ITEM id=82452832b3f6efd5 ===

=== ITEM id=ae195869bd2cfe95 ===
Gold sentences and facts:
Sentence 0: `` Bad Boys '' is a 1987 song by the Jamaican reggae band Inner Circle , which obtained high popularity in the United States after its re-release in 1993 , peaking at number 8 on the Billboard Hot 100 and number 7 on the Top 40 Mainstream . It is well known as the opening theme to the U.S. TV show Cops .
  Fact 0.0: Bad Boys is a 1987 song.
  Fact 0.1: Bad Boys is a song by the Jamaican reggae band Inner Circle.
  Fact 0.2: Bad Boys obtained high popularity in the United States after its re-release in 1993.
  Fact 0.3: Bad Boys peaked at number 8 on the Billboard Hot 100.
  Fact 0.4: Bad Boys peaked at number 7 on the Top 40 Mainstream.
  Fact 0.5: Bad Boys is well known as the opening theme to the U.S. TV show Cops.

Retrieved items:
<item 1>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 1>

<item 2>
Bad Boys (Inner Circle song)
Chart performance ( edit ) Weekly charts ( edit ) Year Chart Peak position 1991 Finland ( Suomen virallinen lista ) Norway ( VG - lista ) Sweden (
Sverigetopplistan ) Australia ( ARIA ) 25 Austria ( Ö3 Austria Top 40 ) 21 Canada Top Singles ( RPM ) 19 Germany ( Official German Charts ) 35 Ireland ( IRMA )
26 New Zealand ( Recorded Music NZ ) 5 UK Singles ( Official Charts Company ) 52 US Billboard Hot 100 8 US Billboard Hot Dance Music / Maxi - Singles Sales 15
US Billboard Hot R&B Singles 58 US Billboard Rhythmic Top 40 27 US Billboard Top 40 Mainstream 7 2006 US Billboard Hot Ringtones 10 2007 US Billboard Hot
Ringtones 20 Year - end charts ( edit ) Chart ( 1993 ) Rank New Zealand ( Recorded Music NZ ) 15 US Billboard Hot 100 69 In popular culture ( edit ) `` Bad Boys
'' was selected as the theme song to Cops because a field producer for the show happened to be a fan of Inner Circle .
</item 2>

<item 3>
Jack McCoy
( hide ) Law & Order Characters Police Sgt . Max Greevey Sgt . Phil Cerreta Lennie Briscoe Joe Fontana Ed Green Cyrus Lupo Mike Logan Rey Curtis Nick Falco Nina
Cassady Kevin Bernard Captain Donald Cragen Lieutenant Anita Van Buren District attorneys Benjamin Stone Jack McCoy Michael Cutter Paul Robinette Claire Kincaid
Jamie Ross Abbie Carmichael Serena Southerlyn Alexandra Borgia Connie Rubirosa Adam Schiff Nora Lewin Arthur Branch Recurring characters Elizabeth Olivet , Ph.
D . Emil Skoda , M.D. Elizabeth Rodgers , M.D.
</item 3>

<item 4>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 4>

<item 5>
Law & Order: Special Victims Unit
Commanding officers Donald Cragen Olivia Benson Assistant District Attorneys Alexandra Cabot Casey Novak Kim Greylek Rafael Barba Doctors George Huang Melinda
Warner Other characters Barry Moredock Brian Cassidy Elizabeth Donnelly Michael Cutter Sonya Paxton Jack McCoy Seasons Episodes Season 1 5 6 7 8 9 10 11 12 13
14 15 16 17 18 19 Notable episodes `` Payback '' `` 911 '' `` Authority '' `` Cold '' `` Zebras '' `` Behave '' `` Smoked '' `` Scorched Earth '' `` Justice
Denied '' `` Father Dearest '' `` Rhodium Nights '' `` Lost Reputation '' `` Above Suspicion '' `` Acceptable Loss '' `` Manhattan Vigil '' `` Comic Perversion
'' `` Glasgowman 's Wrath '' `` Chicago Crossover '' See also Cast members Awards and nominations Law & Order franchise Chicago franchise Chicago Fire
Characters Matthew Casey Kelly Severide Episodes Season 1 5 6 Chicago P.D.
</item 5>

<item 6>
Blue Bloods (season 5)
Danny hears a waitress who sounds like one of the robbers . He later visits the woman 's husband and find he has no legs and is also suffering from traumatic
brain injury the result of combat . He tells the detectives he has had numerous delays in approval for treatment from the Veteran 's Administration , but that
his wife was able to come up with $100,000 in a `` fundraiser '' . Danny and Maria realize the bank robbery was the `` fundraiser '' , but they do not let on
the soldier . Danny also finds out the woman was cheating on her husband , he gets photographic proof and he uses it as leverage to get her to confess . Each
member of the group ultimately receives five years in prison , while Danny and Frank are able to pull some strings with the VA to get the soldier treated
immediately . Meanwhile , the work of a guerilla street artist named Spanky ( in the vein of Banksy ) is causing public scares in the form of pop - ups that
emerge from unattended bags . This creates a rift between Garrett and Gormley on how to treat these actions while maintaining the image of the NYPD .
</item 6>

<item 7>
Stranger Things
The show 's theme is based on an unused work Stein composed much earlier that ended up in the library of work they shared with the production staff , who
thought that with some reworking would be good for the opening credits . In addition to original music , Stranger Things features period music from artists
including The Clash , Joy Division , Toto , New Order , The Bangles , Foreigner , Echo and the Bunnymen , Peter Gabriel , and Corey Hart , as well as excerpts
from Tangerine Dream , John Carpenter , and Vangelis . In particular , The Clash 's `` Should I Stay or Should I Go '' was specifically picked to play at
pivotal moments of the story , such as when Will is trying to communicate with Joyce from the Upside Down . Soundtracks Stranger Things ' original soundtrack ,
consisting of 75 songs from Stein and Dixon split across two volumes , was released by Lakeshore Records . Digital release and streaming options were released
on August 10 and 19 , 2016 for the two volumes , respectively , while retail versions were available on September 16 and 23 , 2016 .
</item 7>

<item 8>
Jack McCoy
Episode 8 . November 29 , 1995 . NBC . Jump up ^ `` Thrill '' . Law & Order . Season 8 . Episode 1 . September 24 , 1997 . NBC . Jump up ^ `` Under God '' . Law
& Order . Season 13 . Episode 12 . February 5 , 2003 . NBC . Jump up ^ `` Listing for `` Confidential '' `` . The Futon Critic ( Press release ) . NBC Universal
. February 19 , 2010 . ^ Jump up to : Tucker , Ken . `` TV : Sam Waterston 's bark keeps giving Law & Order its bite '' , Entertainment Weekly 1026 ( December
19 , 2008 ) : 49 . ( hide ) Law & Order Characters Police Sgt . Max Greevey Sgt .
</item 8>

<item 9>
Like a Boss
Banks , however , claims that he has yet to see any sort of a royalty for the song . Another track from the duo was later featured on the band 's 2011 album
Turtleneck & Chain in the song `` Shy Ronnie 2 : Ronnie & Clyde . '' Censorship ( edit ) This section does not cite any sources . Please help improve this
section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2015 ) ( Learn how and when to remove this
template message ) On the NBC version of this digital short , the part where Andy has a gun in his mouth and whines , `` Oh fuck , man , I ca n't fuckin ' do it
. Shit ! '' had the shot of Andy with the gun in his mouth shortened and redubbed the line to , `` Oh crap , man , I ca n't effin ' do it . Ugh ! ! ''
References ( edit ) Jump up ^ Van Dril , Eric ( 5 April 2009 ) . `` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' .
</item 9>

<item 10>
Bad Boys (Inner Circle song)
It is well known as the opening theme to the U.S. TV show Cops . Contents ( hide ) 1 History 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 In
popular culture 4 See also 5 References 6 External links History ( edit ) The song was originally released in 1987 on the album One Way and obtained minor
airplay and poor sales . However , in 1992 , it was also included on the Bad to the Bone album , and the song was released as a single in 1993 , after the
unexpected success of their previous international hit single `` Sweat ( A La La La La Long ) '' . `` Bad Boys '' reached number 52 in the United Kingdom and
number 8 in the United States ; in the latter territory , it was certified Gold and sold 600,000 copies . The song was also released as B - side on Inner Circle
's 1992 single `` Sweat ( A La La La La Long ) '' . The song was also released as a single in mid-1991 in Norway ; it peaked at number one for three weeks .
</item 10>
=== END ITEM id=ae195869bd2cfe95 ===
