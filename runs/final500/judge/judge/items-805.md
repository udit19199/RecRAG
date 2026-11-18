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

This file has 6 items.

=== ITEM id=4dc46fb646fdaedc ===
Gold sentences and facts:
Sentence 0: Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out .
  Fact 0.0: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 0.1: Jane and Rafael's plan to buy the apartment falls apart when Mateo breaks an antique $4,000 clock.
  Fact 0.2: Rafael moves into the Villanueva house.
  Fact 0.3: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 0.4: Rafael moves back out of the Villanueva house after the argument.
  Fact 0.5: Xiomara starts chemotherapy.
  Fact 0.6: Xiomara makes a friend named Donna.
  Fact 0.7: The timeline moves forward nine weeks.
  Fact 0.8: Donna dies of unforeseen consequences.
  Fact 0.9: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 0.10: Alba and Rafael are still in a fight.
  Fact 0.11: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 0.12: Alba breaks down to Jane.
  Fact 0.13: Alba and Rafael make up.
  Fact 0.14: Xiomara bonds with River over telenovelas.
  Fact 0.15: Xiomara accidentally eats a pot brownie.
  Fact 0.16: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 0.17: Jane Ramos gets herself disbarred.
  Fact 0.18: As a result of Jane Ramos being disbarred, the case against Petra is thrown out.
Sentence 1: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 65 `` Chapter Sixty - Five '' Brad Silberling Jennie Snyder Urman & Paul Sciarrotta October 13 , 2017 ( 2017 - 10 - 13 ) 0.68 Picking up immediately after where Season 3 left off , Jane is reunited with her first love , Adam , who found the love letter that Michael wrote . She is surprised , but happy , to see him . However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom . However , when Adam gets a job opportunity out of state , he asks for Jane 's opinion on if he should stay or go . Meanwhile , Rogelio continues to face issues with his co-star at work and things are about to get even worse . Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby . Also , Petra and Rafael begin working together to try and buy back the hotel from Luisa , but they are running into trouble involving Anezka , causing tension between them . 67 `` Chapter Sixty - Seven '' Fernando Sariñana Carolina Rivera & Micah Schraft October 27 , 2017 ( 2017 - 10 - 27 ) 0.60 Jane and Adam 's relationship seems to be moving on , and Jane thinks that it might be time for him to meet her son Mateo , but Rafael is not happy with this idea . Meanwhile , as the arrival of Darci 's baby approaches , Rogelio 's jealously of her new boyfriend increases as well . Trying to be supportive , Xo encourages Rogelio to step up and be the father he was meant to be . Elsewhere , Rafael is determined to get his hotel and life back from Luisa and will do whatever he has to do to make it happen , with or without Petra 's help . 68 `` Chapter Sixty - Eight '' Gina Lamar Deirdre Shaw & Chantelle M. Wells November 3 , 2017 ( 2017 - 11 - 03 ) 0.69 Jane becomes angry when she learns that Rafael introduced Mateo to his new girlfriend , Katherine , without her consent because she thinks Katherine is a negative influence on both Rafael and Mateo . Jane turns to Petra for help with the situation . Meanwhile , Rogelio and Darci struggle to choose a name for the baby . Elsewhere , Xo , who is focused on opening her own dance studio , decides to go into business with an unlikely foe . 69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not . Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy ) done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day . Meanwhile , Rafael gets a new job , but it is not exactly what he was hoping for . Also , Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa , but Rafael wants nothing to do with the scheme . 71 7 `` Chapter Seventy - One '' Micah Schraft Valentina Garza & Deidre Shaw December 8 , 2017 ( 2017 - 12 - 08 ) 0.65 Anezka is revealed to have faked her death to see where her loyalties lie ; initially , she trusts Petra after believing she saw her cry , but after finding out she was n't crying because of her fake death , she threatens Petra . Later , Anezka falls from Petra 's apartment to her death ... for real . Meanwhile , Xiomara and Jane set off on Jane 's book tour , when Alba decides to tag along . After they get back , Jane and Rafael share a kiss . 72 8 `` Chapter Seventy - Two '' Melanie Mayron Carolina Rivera & Paul Sciarrotta January 26 , 2018 ( 2018 - 01 - 26 ) 0.68 Mateo starts kindergarten and invites a friend over to Xiomara and Rogelio 's house . Jane and Rafael try to cover up that they are pretending to live in their house so they do n't get kicked out . Jane and Rafael show mixed emotions following the kiss and try again , which Mateo notices . He then proclaims : `` Does this mean you 're going to get married ? '' After Xo convinces Rogelio to attend couples therapy , they both are surprised by their emotions . Meanwhile , Petra hires a lawyer named Jane Ramos ( Rosario Dawson ) , to ensure she gets no charges pending Anezka 's death . Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 , 2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date . Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow . Rogelio is convinced he has male post-partum depression , which puts him at odds with with celebrity Post-Partum Advocate River Fields ( guest star Brooke Shields ) on The Talk . Also , Alba finally confides in Jane why she turned down the marriage proposal , leaving Jane to help Alba in the romance department when Jane takes Alba to a sex store and buys her a vibrator . Elsewhere , Jane and Rafael agree to have sex , and Petra questions her own sexual identity when she starts to have sex dreams about Jane `` J.R. '' Ramos after the lawyer begins flirting with her . 75 11 `` Chapter Seventy - Five '' Gina Lamar Merigan Mulhern March 2 , 2018 ( 2018 - 03 - 02 ) 0.59 Jane is ready to move on from the Marbella Hotel issues and pursue writing full time , but an unexpected development forces her to consider returning to her teaching roots . Xiomara decides to go back to school where she listens in on an art history class , while Jane teaches a seminar about writing . Here , Jane finds out Jonathan Chavez , her former advisor , is taking advantage of another student of his . While trying to warn the student , Jane accidentally grabs her breasts to avoid falling over and , as a result , does n't get the teaching job she was applying for . Meanwhile , Petra gets arrested for Anezka 's murder due to the blackmailer planting the screws in her apartment , leading Jane Ramos to tell Petra that she was framing her because she was being threatened by an unknown blackmailer . Also , Rafael and Jane continue their affair . Elsewhere , Rogelio tries to get Eva Longoria to join his Passions of Santos reboot , but instead finds out the two are distant cousins . 76 12 `` Chapter Seventy - Six '' Melanie Mayron Leah Longoria March 9 , 2018 ( 2018 - 03 - 09 ) 0.58 Jane 's fixation on a negative review has left her with writer 's block , so Rogelio suggests she take a writing class to help . With the encouragement of Jane , Rafael is ready to dig into his past to find out who his real biological parents are . Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 ) 0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind . Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R 's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65 Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider . Alba is frustrated with Rogelio when it seems he is once again making his career a priority over Xo 's well being . Meanwhile , Petra turns to Rafael for advice on her love life after he finally learns about her new romance with J.R. 79 15 `` Chapter Seventy - Nine '' Gina Lamar Paul Sciarrotta April 6 , 2018 ( 2018 - 04 - 06 ) 0.53 80 16 `` Chapter Eighty '' Micah Schraft Micah Schraft April 13 , 2018 ( 2018 - 04 - 13 ) 0.57 Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out . 81 17 `` Chapter Eighty - One '' Gina Lamar Jennie Snyder Urman & Paul Sciarrotta April 20 , 2018 ( 2018 - 04 - 20 ) 0.58 Rafael intends to propose to Jane , however Rose 's news unsettles him . Alba becomes a citizen and the Villanueva 's throw her a surprise party . Jane Ramos and Petra plan to move in together , until Jane finds out that Petra killed Anezka . As she walks out , Krishna tells her that her life is in danger . J.R. saves Petra 's life by shooting , and killing , the blackmailer . As Jane meets Rafael , Michael is revealed to be alive .
  Fact 1.0: The listing has columns for overall episode number, season episode number, title, director, writers, original air date, and U.S. viewers in millions.
  Fact 1.1: Chapter Sixty-Five is episode 65 overall, directed by Brad Silberling and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.2: Chapter Sixty-Five first aired October 13, 2017, with 0.68 million U.S. viewers.
  Fact 1.3: Chapter Sixty-Five picks up immediately after where Season 3 left off.
  Fact 1.4: Jane is reunited with her first love, Adam, who found the love letter that Michael wrote.
  Fact 1.5: Jane is surprised but happy to see Adam.
  Fact 1.6: Xo and Alba are less thrilled that Adam is back.
  Fact 1.7: Rafael loses ownership of the hotel to his sister.
  Fact 1.8: As a result, Rafael finds himself living at Jane's house, broke and homeless.
  Fact 1.9: Jane and Rafael learn that the missing Petra has been found after she got away from Anezka.
  Fact 1.10: Jane and Rafael rush to Petra's side, but Anezka is nowhere to be found.
  Fact 1.11: When Petra returns to the hotel, Luisa thinks Petra is Anezka.
  Fact 1.12: Petra goes along with the game until the real Anezka returns.
  Fact 1.13: Xo and Rogelio's married life is interrupted when they confront the realization that Darci is having Rogelio's baby.
  Fact 1.14: Darci wants full custody of the baby.
  Fact 1.15: Chapter Sixty-Six is directed by Gina Lamar and written by Valentina Garza, Jessica O'Toole and Amy Rardin.
  Fact 1.16: Chapter Sixty-Six first aired October 20, 2017, with 0.61 million U.S. viewers.
  Fact 1.17: Jane's relationship with Adam helps bring out her 'fun side'.
  Fact 1.18: Adam's relationship with Jane reminds her that she is not just a mom.
  Fact 1.19: Adam gets a job opportunity out of state.
  Fact 1.20: Adam asks Jane's opinion on whether he should stay or go.
  Fact 1.21: Rogelio continues to face issues with his co-star at work.
  Fact 1.22: Rogelio's issues with his co-star are about to get even worse.
  Fact 1.23: Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby.
  Fact 1.24: Petra and Rafael begin working together to try to buy back the hotel from Luisa.
  Fact 1.25: Petra and Rafael run into trouble involving Anezka.
  Fact 1.26: Petra and Rafael's trouble involving Anezka causes tension between them.
  Fact 1.27: Chapter Sixty-Seven is directed by Fernando Sariñana and written by Carolina Rivera and Micah Schraft.
  Fact 1.28: Chapter Sixty-Seven first aired October 27, 2017, with 0.60 million U.S. viewers.
  Fact 1.29: Jane and Adam's relationship seems to be moving on.
  Fact 1.30: Jane thinks it might be time for Adam to meet her son Mateo.
  Fact 1.31: Rafael is not happy with the idea of Adam meeting Mateo.
  Fact 1.32: As the arrival of Darci's baby approaches, Rogelio's jealousy of Darci's new boyfriend increases.
  Fact 1.33: Xo encourages Rogelio to step up and be the father he was meant to be.
  Fact 1.34: Rafael is determined to get his hotel and life back from Luisa.
  Fact 1.35: Rafael will do whatever he has to do to get his hotel and life back, with or without Petra's help.
  Fact 1.36: Chapter Sixty-Eight is directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.37: Chapter Sixty-Eight first aired November 3, 2017, with 0.69 million U.S. viewers.
  Fact 1.38: Jane becomes angry when she learns Rafael introduced Mateo to his new girlfriend, Katherine, without her consent.
  Fact 1.39: Jane thinks Katherine is a negative influence on both Rafael and Mateo.
  Fact 1.40: Jane turns to Petra for help with the situation.
  Fact 1.41: Rogelio and Darci struggle to choose a name for the baby.
  Fact 1.42: Xo is focused on opening her own dance studio.
  Fact 1.43: Xo decides to go into business with an unlikely foe.
  Fact 1.44: Chapter Sixty-Nine is episode 69 overall and season 5, directed by Stuart Gillard and written by Paul Sciarrotta.
  Fact 1.45: Chapter Sixty-Nine first aired November 10, 2017, with 0.65 million U.S. viewers.
  Fact 1.46: After Adam reveals his dating past to Jane, Jane becomes unsure and insecure.
  Fact 1.47: Jane's insecurity puts a strain on her relationship with Adam.
  Fact 1.48: Lina seeks Jane's advice about whether she should get married.
  Fact 1.49: Petra hears important information that could help Rafael get the hotel back.
  Fact 1.50: Xo asks Rogelio to consider having a vasectomy.
  Fact 1.51: Rogelio is very hesitant about the vasectomy.
  Fact 1.52: Chapter Seventy is episode 70 overall and season 6, directed by Melanie Mayron and written by Carolina Rivera and Micah Schraft.
  Fact 1.53: Chapter Seventy first aired November 17, 2017, with 0.61 million U.S. viewers.
  Fact 1.54: Jane's book is finally being released.
  Fact 1.55: Jane's plans to have a party at the local bookshop where she was inspired to be a writer hit a snag.
  Fact 1.56: Rogelio devises a plan to save the day for Jane.
  Fact 1.57: One of Rogelio's arch-rivals threatens to ruin the day.
  Fact 1.58: Rafael gets a new job, but it is not exactly what he was hoping for.
  Fact 1.59: Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa.
  Fact 1.60: Rafael wants nothing to do with Petra's scheme.
  Fact 1.61: Chapter Seventy-One is episode 71 overall and season 7, directed by Micah Schraft and written by Valentina Garza and Deidre Shaw.
  Fact 1.62: Chapter Seventy-One first aired December 8, 2017, with 0.65 million U.S. viewers.
  Fact 1.63: Anezka is revealed to have faked her death to see where her loyalties lie.
  Fact 1.64: Anezka initially trusts Petra after believing she saw Petra cry.
  Fact 1.65: Anezka threatens Petra after finding out Petra was not crying because of Anezka's fake death.
  Fact 1.66: Later, Anezka falls from Petra's apartment to her death, for real.
  Fact 1.67: Xiomara and Jane set off on Jane's book tour.
  Fact 1.68: Alba decides to tag along on Jane's book tour.
  Fact 1.69: After they get back, Jane and Rafael share a kiss.
  Fact 1.70: Chapter Seventy-Two is episode 72 overall and season 8, directed by Melanie Mayron and written by Carolina Rivera and Paul Sciarrotta.
  Fact 1.71: Chapter Seventy-Two first aired January 26, 2018, with 0.68 million U.S. viewers.
  Fact 1.72: Mateo starts kindergarten.
  Fact 1.73: Mateo invites a friend over to Xiomara and Rogelio's house.
  Fact 1.74: Jane and Rafael try to cover up that they are pretending to live in their house so they don't get kicked out.
  Fact 1.75: Jane and Rafael show mixed emotions following the kiss and try again, which Mateo notices.
  Fact 1.76: Mateo proclaims: 'Does this mean you're going to get married?'
  Fact 1.77: After Xo convinces Rogelio to attend couples therapy, Xo and Rogelio are both surprised by their emotions.
  Fact 1.78: Petra hires a lawyer named Jane Ramos, played by Rosario Dawson.
  Fact 1.79: Petra hires Jane Ramos to ensure Petra gets no charges pending Anezka's death.
  Fact 1.80: Rafael moves out of Jane's house into his new apartment.
  Fact 1.81: Chapter Seventy-Three is episode 73 overall and season 9, directed by Eric Lea and written by Valentina Garza and Chantelle M. Wells.
  Fact 1.82: Chapter Seventy-Three first aired February 2, 2018, with 0.67 million U.S. viewers.
  Fact 1.83: Jane and Rafael realize that the uncertain state of their relationship is stressing out Mateo.
  Fact 1.84: Jane and Rafael decide that they need to make a decision.
  Fact 1.85: Petra is on the outs with Jane and Rafael after the Luisa fiasco.
  Fact 1.86: Petra tries to figure out a way to get back into Jane and Rafael's good graces.
  Fact 1.87: Rogelio and Xo are both contemplating their career choices.
  Fact 1.88: Chapter Seventy-Four is episode 74 overall and season 10, directed by Gina Rodriguez and written by Micah Schraft and Paul Sciarrotta.
  Fact 1.89: Chapter Seventy-Four first aired February 9, 2018, with 0.80 million U.S. viewers.
  Fact 1.90: Jane and Rafael try to be friends, which works great until Jane goes on a date.
  Fact 1.91: Petra and Jane's new partnership is off to a rocky start.
  Fact 1.92: Petra and Jane both know they have to find a way to work together.
  Fact 1.93: Rogelio is convinced he has male post-partum depression.
  Fact 1.94: Rogelio's belief puts him at odds with celebrity Post-Partum Advocate River Fields on The Talk.
  Fact 1.95: River Fields is guest starred by Brooke Shields.
  Fact 1.96: Alba finally confides in Jane why she turned down the marriage proposal.
  Fact 1.97: Jane helps Alba in the romance department.
  Fact 1.98: Jane takes Alba to a sex store and buys her a vibrator.
  Fact 1.99: Jane and Rafael agree to have sex.
  Fact 1.100: Petra questions her own sexual identity.
  Fact 1.101: Petra starts to have sex dreams about Jane 'J.R.' Ramos after the lawyer begins flirting with her.
  Fact 1.102: Chapter Seventy-Five is episode 75 overall and season 11, directed by Gina Lamar and written by Merigan Mulhern.
  Fact 1.103: Chapter Seventy-Five first aired March 2, 2018, with 0.59 million U.S. viewers.
  Fact 1.104: Jane is ready to move on from the Marbella Hotel issues and pursue writing full time.
  Fact 1.105: An unexpected development forces Jane to consider returning to her teaching roots.
  Fact 1.106: Xiomara decides to go back to school, where she listens in on an art history class.
  Fact 1.107: Jane teaches a seminar about writing.
  Fact 1.108: Jane finds out Jonathan Chavez, her former advisor, is taking advantage of another student.
  Fact 1.109: While trying to warn the student, Jane accidentally grabs her breasts to avoid falling over.
  Fact 1.110: Because of this, Jane does not get the teaching job she was applying for.
  Fact 1.111: Petra gets arrested for Anezka's murder due to the blackmailer planting the screws in Petra's apartment.
  Fact 1.112: Jane Ramos tells Petra that she was framing Petra because Jane was being threatened by an unknown blackmailer.
  Fact 1.113: Rafael and Jane continue their affair.
  Fact 1.114: Rogelio tries to get Eva Longoria to join his Passions of Santos reboot.
  Fact 1.115: Rogelio finds out that he and Eva Longoria are distant cousins.
  Fact 1.116: Chapter Seventy-Six is episode 76 overall and season 12, directed by Melanie Mayron and written by Leah Longoria.
  Fact 1.117: Chapter Seventy-Six first aired March 9, 2018, with 0.58 million U.S. viewers.
  Fact 1.118: Jane's fixation on a negative review has left her with writer's block.
  Fact 1.119: Rogelio suggests Jane take a writing class to help.
  Fact 1.120: With Jane's encouragement, Rafael is ready to dig into his past to find out who his real biological parents are.
  Fact 1.121: Rogelio is trying to make things less about him and more about Xo.
  Fact 1.122: Rogelio grows suspicious when he learns Xo is keeping a secret from him.
  Fact 1.123: The secret Xo has been keeping is not what Rogelio suspected.
  Fact 1.124: Petra tests the loyalty of her lawyer, Jane Ramos, by pretending to be romantically interested in her.
  Fact 1.125: Petra is trying to find the identity of the blackmailer.
  Fact 1.126: Petra confides in Jane R. about her feelings, which are not fake.
  Fact 1.127: Petra and Jane R. sleep together for the first time.
  Fact 1.128: Chapter Seventy-Seven is episode 77 overall and season 13, directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.129: Chapter Seventy-Seven first aired March 16, 2018, with 0.59 million U.S. viewers.
  Fact 1.130: Jane, Rogelio and Alba learn that Xo might have breast cancer.
  Fact 1.131: Jane, Rogelio and Alba try to comfort Xo.
  Fact 1.132: Xo insists they let her live her life.
  Fact 1.133: Rogelio finally gets River Fields to agree to meet with him.
  Fact 1.134: Rogelio learns that River has another plan in mind.
  Fact 1.135: Rafael is on a mission to find his sister, who may have information he has been looking for about his origins.
  Fact 1.136: Rafael visits Rose in prison to ask for her help.
  Fact 1.137: Rafael quickly turns Rose down when she demands to know Luisa's location.
  Fact 1.138: Jane is upset with Petra after Jane's twin girls spill a secret to Mateo about the Tooth Fairy being make-believe.
  Fact 1.139: Jane R. Ramos tells Petra she is not interested in a relationship after their romantic night together.
  Fact 1.140: The blackmailer paid a visit to Jane R.'s dementia-stricken mother in a rest home.
  Fact 1.141: The identity of Jane R.'s blackmailer is finally revealed at the end of the episode.
  Fact 1.142: Chapter Seventy-Eight is episode 78 overall and season 14, directed by Justin Baldoni and written by Valentina Garza and Micah Schraft.
  Fact 1.143: Chapter Seventy-Eight first aired March 23, 2018, with 0.65 million U.S. viewers.
  Fact 1.144: Xo struggles to make an important decision on whether or not to get a full mastectomy.
  Fact 1.145: Xo looks to Jane for guidance on the mastectomy decision.
  Fact 1.146: Xo's decision leaves Rogelio feeling like an outsider.
  Fact 1.147: Alba is frustrated with Rogelio when he once again seems to make his career a priority over Xo's well-being.
  Fact 1.148: Petra turns to Rafael for advice on her love life.
  Fact 1.149: Rafael finally learns about Petra's new romance with J.R.
  Fact 1.150: Chapter Seventy-Nine is episode 79 overall and season 15, directed by Gina Lamar and written by Paul Sciarrotta.
  Fact 1.151: Chapter Seventy-Nine first aired April 6, 2018, with 0.53 million U.S. viewers.
  Fact 1.152: Chapter Eighty is episode 80 overall and season 16, directed by Micah Schraft and written by Micah Schraft.
  Fact 1.153: Chapter Eighty first aired April 13, 2018, with 0.57 million U.S. viewers.
  Fact 1.154: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 1.155: Jane and Rafael's plan falls apart when Mateo breaks an antique $4,000 clock.
  Fact 1.156: Rafael moves into the Villanueva house.
  Fact 1.157: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 1.158: Rafael moves back out after the argument.
  Fact 1.159: Xiomara starts chemotherapy.
  Fact 1.160: Xiomara makes a friend named Donna.
  Fact 1.161: The timeline moves forward nine weeks.
  Fact 1.162: Donna dies of unforeseen consequences.
  Fact 1.163: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 1.164: Alba and Rafael are still in a fight.
  Fact 1.165: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 1.166: Alba breaks down to Jane.
  Fact 1.167: Alba and Rafael make up.
  Fact 1.168: Xiomara bonds with River over telenovelas.
  Fact 1.169: Xiomara accidentally eats a pot brownie.
  Fact 1.170: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 1.171: Jane Ramos gets herself disbarred.
  Fact 1.172: As a result, Jane Ramos gets the case against Petra thrown out.
  Fact 1.173: Chapter Eighty-One is episode 81 overall and season 17, directed by Gina Lamar and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.174: Chapter Eighty-One first aired April 20, 2018, with 0.58 million U.S. viewers.
  Fact 1.175: Rafael intends to propose to Jane.
  Fact 1.176: Rose's news unsettles Rafael.
  Fact 1.177: Alba becomes a citizen.
  Fact 1.178: The Villanueva family throws Alba a surprise party.
  Fact 1.179: Jane Ramos and Petra plan to move in together.
  Fact 1.180: Jane finds out Petra killed Anezka.
  Fact 1.181: As Jane walks out, Krishna tells her that her life is in danger.
  Fact 1.182: J.R. saves Petra's life by shooting and killing the blackmailer.
  Fact 1.183: As Jane meets Rafael, Michael is revealed to be alive.

Retrieved items:
<item 1>
Jane the Virgin (season 4)
Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for
her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret
to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not
interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R
's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65
Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider
.
</item 1>

<item 2>
Jane the Virgin (season 4)
67 0.3 0.60 0.6 1.27 10 `` Chapter Seventy - Four '' February 9 , 2018 0.3 / 1 0.80 0.3 0.59 0.6 1.39 11 `` Chapter Seventy - Five '' March 2 , 2018 0.2 / 1
0.59 0.3 0.62 0.5 1.21 12 `` Chapter Seventy - Six '' March 9 , 2018 0.2 / 1 0.58 0.3 0.53 0.5 1.12 13 `` Chapter Seventy - Seven '' March 16 , 2018 0.2 / 1
0.59 0.3 0.59 0.5 1.19 14 `` Chapter Seventy - Eight '' March 23 , 2018 0.2 / 1 0.65 0.3 N / A 0.5 N / A 15 `` Chapter Seventy - Nine '' April 6 ,
</item 2>

<item 3>
Jane the Virgin (season 4)
69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she
becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not .
Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy )
done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's
book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help
Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day .
</item 3>

<item 4>
Jane the Virgin (season 4)
Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But
the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically
interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of
them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 )
0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live
her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind .
</item 4>

<item 5>
Jane the Virgin (season 4)
Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 ,
2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they
decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to
get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez
Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date .
Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow .
</item 5>

<item 6>
Jane the Virgin (season 4)
Trying to be supportive , Xo encourages Rogelio to step up and be the father he was meant to be . Elsewhere , Rafael is determined to get his hotel and life
back from Luisa and will do whatever he has to do to make it happen , with or without Petra 's help . 68 `` Chapter Sixty - Eight '' Gina Lamar Deirdre Shaw &
Chantelle M. Wells November 3 , 2017 ( 2017 - 11 - 03 ) 0.69 Jane becomes angry when she learns that Rafael introduced Mateo to his new girlfriend , Katherine ,
without her consent because she thinks Katherine is a negative influence on both Rafael and Mateo . Jane turns to Petra for help with the situation . Meanwhile
, Rogelio and Darci struggle to choose a name for the baby . Elsewhere , Xo , who is focused on opening her own dance studio , decides to go into business with
an unlikely foe .
</item 6>

<item 7>
Jane the Virgin (season 4)
Meanwhile , Rafael gets a new job , but it is not exactly what he was hoping for . Also , Petra has come up with an interesting strategy to get the Marbella
Hotel back from Luisa , but Rafael wants nothing to do with the scheme . 71 7 `` Chapter Seventy - One '' Micah Schraft Valentina Garza & Deidre Shaw December 8
, 2017 ( 2017 - 12 - 08 ) 0.65 Anezka is revealed to have faked her death to see where her loyalties lie ; initially , she trusts Petra after believing she saw
her cry , but after finding out she was n't crying because of her fake death , she threatens Petra . Later , Anezka falls from Petra 's apartment to her death
... for real . Meanwhile , Xiomara and Jane set off on Jane 's book tour , when Alba decides to tag along . After they get back , Jane and Rafael share a kiss .
</item 7>

<item 8>
Jane the Virgin (season 4)
However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself
living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her
side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real
Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants
full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's
relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom .
</item 8>

<item 9>
Jane the Virgin (season 4)
2017 0.2 / 1 0.69 0.3 0.55 0.5 1.24 5 `` Chapter Sixty - Nine '' November 10 , 2017 0.3 / 1 0.65 TBD 0.61 TBD 1.26 6 `` Chapter Seventy '' November 17 , 2017
0.2 / 1 0.61 0.3 0.55 0.5 1.16 7 `` Chapter Seventy - One '' December 8 , 2017 0.3 / 1 0.65 0.3 0.61 0.6 1.25 8 `` Chapter Seventy - Two '' January 26 , 2018
0.2 / 1 0.68 0.4 0.71 0.6 1.39 9 `` Chapter Seventy - Three '' February 2 , 2018 0.3 / 1 0.67 0.3 0.60 0.6 1.
</item 9>

<item 10>
Jane the Virgin (season 4)
Alba is frustrated with Rogelio when it seems he is once again making his career a priority over Xo 's well being . Meanwhile , Petra turns to Rafael for advice
on her love life after he finally learns about her new romance with J.R. 79 15 `` Chapter Seventy - Nine '' Gina Lamar Paul Sciarrotta April 6 , 2018 ( 2018 -
04 - 06 ) 0.53 80 16 `` Chapter Eighty '' Micah Schraft Micah Schraft April 13 , 2018 ( 2018 - 04 - 13 ) 0.57 Jane and Rafael start saving up to buy a one -
bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo ,
the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine
weeks , and Donna dies of unforeseen consequences .
</item 10>
=== END ITEM id=4dc46fb646fdaedc ===

=== ITEM id=244937a027d78b8a ===
Gold sentences and facts:
Sentence 0: The Dead Sea ( Hebrew : יָם הַמֶּלַח ‬ Yam ha - Melah lit . Sea of Salt ; Arabic : البحر الميت ‎ Al - Bahr al - Mayyit ) is a salt lake bordered by Jordan to the east and Israel and Palestine to the west . Its surface and shores are 430.5 metres ( 1,412 ft ) below sea level , Earth 's lowest elevation on land . The Dead Sea is 304 m ( 997 ft ) deep , the deepest hypersaline lake in the world . With a salinity of 342 g / kg , or 34.2 % , ( in 2011 ) , it is 9.6 times as salty as the ocean and one of the world 's saltiest bodies of water . This salinity makes for a harsh environment in which plants and animals can not flourish , hence its name . The Dead Sea is 50 kilometres ( 31 mi ) long and 15 kilometres ( 9 mi ) wide at its widest point . It lies in the Jordan Rift Valley , and its main tributary is the Jordan River .
  Fact 0.0: The Dead Sea is also called יָם הַמֶּלַח (Yam ha-Melah) in Hebrew, which means Sea of Salt.
  Fact 0.1: The Dead Sea is also called البحر الميت (Al-Bahr al-Mayyit) in Arabic.
  Fact 0.2: The Dead Sea is a salt lake.
  Fact 0.3: The Dead Sea is bordered by Jordan to the east.
  Fact 0.4: The Dead Sea is bordered by Israel and Palestine to the west.
  Fact 0.5: The surface and shores of the Dead Sea are 430.5 metres (1,412 ft) below sea level.
  Fact 0.6: The surface and shores of the Dead Sea are Earth's lowest elevation on land.
  Fact 0.7: The Dead Sea is 304 m (997 ft) deep.
  Fact 0.8: The Dead Sea is the deepest hypersaline lake in the world.
  Fact 0.9: In 2011, the Dead Sea had a salinity of 342 g/kg, or 34.2%.
  Fact 0.10: The Dead Sea is 9.6 times as salty as the ocean.
  Fact 0.11: The Dead Sea is one of the world's saltiest bodies of water.
  Fact 0.12: The Dead Sea's salinity makes for a harsh environment in which plants and animals cannot flourish.
  Fact 0.13: The Dead Sea's name comes from its harsh environment in which plants and animals cannot flourish.
  Fact 0.14: The Dead Sea is 50 kilometres (31 mi) long.
  Fact 0.15: The Dead Sea is 15 kilometres (9 mi) wide at its widest point.
  Fact 0.16: The Dead Sea lies in the Jordan Rift Valley.
  Fact 0.17: The main tributary of the Dead Sea is the Jordan River.

Retrieved items:
<item 1>
Dead Sea
The Greeks called it Lake Asphaltites ( Attic Greek ἡ Θάλαττα ἀσφαλτῖτης , hē Thálatta asphaltĩtēs , `` the Asphaltite sea '' ) . Geography Satellite photograph
showing the location of the Dead Sea east of the Mediterranean Sea The Dead Sea is an endorheic lake located in the Jordan Rift Valley , a geographic feature
formed by the Dead Sea Transform ( DST ) . This left lateral - moving transform fault lies along the tectonic plate boundary between the African Plate and the
Arabian Plate . It runs between the East Anatolian Fault zone in Turkey and the northern end of the Red Sea Rift offshore of the southern tip of Sinai . It is
here that the Upper Jordan River / Sea of Galilee / Lower Jordan River water system comes to an end . The Jordan River is the only major water source flowing
into the Dead Sea , although there are small perennial springs under and around the Dead Sea , forming pools and quicksand pits along the edges . There are no
outlet streams .
</item 1>

<item 2>
Dead Sea
Along the southwestern side of the lake is a 210 m ( 700 ft ) tall halite formation called `` Mount Sodom '' . Natural History There are two contending
hypotheses about the origin of the low elevation of the Dead Sea . The older hypothesis is that the Dead Sea lies in a true rift zone , an extension of the Red
Sea Rift , or even of the Great Rift Valley of eastern Africa . A more recent hypothesis is that the Dead Sea basin is a consequence of a `` step - over ''
discontinuity along the Dead Sea Transform , creating an extension of the crust with consequent subsidence . Around 3.7 million years ago , what is now the
valley of the Jordan River , Dead Sea , and the northern Wadi Arabah was repeatedly inundated by waters from the Mediterranean Sea . The waters formed in a
narrow , crooked bay that is called by geologists the Sedom Lagoon , which was connected to the sea through what is now the Jezreel Valley . The floods of the
valley came and went depending on long - scale climate change . The Sedom Lagoon deposited beds of salt that eventually became 2.5 km ( 1.55 mi ) thick .
</item 2>

<item 3>
Dead Sea
depth 298 m ( 978 ft ) ( elevation of deepest point , 728 m BSL ( below sea level ) , minus current surface elevation ) Water volume 114 km ( 27 cu mi ) Shore
length 135 km ( 84 mi ) Surface elevation − 430.5 m ( − 1,412 ft ) ( 2016 ) References Shore length is not a well - defined measure . Play media Short video
about the Dead Sea from the Israeli News Company The Dead Sea ( Hebrew : יָם הַמֶּלַח ‬ Yam ha - Melah lit . Sea of Salt ; Arabic : البحر الميت ‎ Al - Bahr al -
Mayyit ) is a salt lake bordered by Jordan to the east and Israel and Palestine to the west . Its surface and shores are 430.5 metres ( 1,412 ft ) below sea
level , Earth 's lowest elevation on land . The Dead Sea is 304 m ( 997 ft ) deep , the deepest hypersaline lake in the world .
</item 3>

<item 4>
Dead Sea
Dead Sea - wikipedia Dead Sea Jump to : navigation , search For the novel , see Dead Sea ( novel ) . Dead Sea ים המלח ‬ ( in Hebrew ) البحر الميت ( in Arabic )
A view of the sea from the Israeli shore Location Israel Jordan Palestine Coordinates 31 ° 30 ′ N 35 ° 30 ′ E ﻿ / ﻿ 31.500 ° N 35.500 ° E ﻿ / 31.500 ; 35.500
Coordinates : 31 ° 30 ′ N 35 ° 30 ′ E ﻿ / ﻿ 31.500 ° N 35.500 ° E ﻿ / 31.500 ; 35.500 Lake type Endorheic Hypersaline Primary inflows Jordan River Primary
outflows None Catchment area 41,650 km ( 16,080 sq mi ) Basin countries Israel Jordan Palestine Max. length 50 km ( 31 mi ) ( northern basin only ) Max . width
15 km ( 9.3 mi ) Surface area 605 km ( 234 sq mi ) ( 2016 ) Average depth 199 m ( 653 ft ) Max .
</item 4>

<item 5>
Dead Sea
In January 2015 it was reported that the level of water is now dropping by 1 m ( 3 ft ) a year . On 27 November 2016 , it was announced that the Jordanian
government is shortlisting five consortiums to implement the project . Jordan 's ministry of Water and Irrigation said that the $100 million first phase of the
project will begin construction in the first quarter of 2018 , and will be completed by 2021 . Views in 1972 , 1989 , and 2011 compared See also Red Sea -- Dead
Sea Water Conveyance Aral Sea Dead Sea Works Great Salt Lake List of drying lakes List of places on land with elevations below sea level Mediterranean -- Dead
Sea Canal Benjamin Elazari Volcani World Discoveries III : Dead Sea References ^ Jump up to : `` Virtual Israel Experience : The Dead Sea '' . Jewish Virtual
Library . Retrieved 21 January 2013 . ^ Jump up to : `` The Dead Sea Is Dying Fast : Is It Too Late to Save It , or Was It Always a Lost Cause ? '' . Haaretz. 7
October 2016 .
</item 5>

<item 6>
Dead Sea
Year Water level ( m ) Surface ( km ) 1930 − 390 1050 1980 − 400 680 − 407 675 − 411 670 − 417 662 − 423 655 2016 − 430.5 605 Sources : Israel Oceanographic and
Limnological Research , Haaretz , Jewish Virtual Library , Jordan Valley Authority . In October 2009 , the Jordanians announced accelerated plans to extract
around 300 million cubic metres ( 11 billion cubic feet ) of water per year from the Red Sea , desalinate it for use as fresh water and send the waste water to
the Dead Sea by tunnel , despite concerns about inadequate time to assess the potential environmental impact . According to Jordan 's minister for water ,
General Maysoun Zu'bi , this project could be considered as the first phase of the Red Sea -- Dead Sea Water Conveyance . In December 2013 , Israel , Jordan and
the Palestinian Authority signed an agreement for laying a water pipeline to link the Red Sea with the Dead Sea . The pipeline will be 180 km ( 110 mi ) long
and is estimated to take up to five years to complete .
</item 6>

<item 7>
Dead Sea
As the water level decreases , the characteristics of the Sea and surrounding region may substantially change . The Dead Sea level drop has been followed by a
groundwater level drop , causing brines that used to occupy underground layers near the shoreline to be flushed out by freshwater . This is believed to be the
cause of the recent appearance of large sinkholes along the western shore -- incoming freshwater dissolves salt layers , rapidly creating subsurface cavities
that subsequently collapse to form these sinkholes . Gully in unconsolidated Dead Sea sediments exposed by recession of water levels . It was excavated by
floods from the Judean Mountains in less than a year . In May 2009 at the World Economic Forum , Jordan announced its plans to construct the `` Jordan National
Red Sea Development Project '' ( JRSP ) . This is a plan to convey seawater from the Red Sea near Aqaba to the Dead Sea . Water would be desalinated along the
route to provide fresh water to Jordan , with the brine discharge sent to the Dead Sea for replenishment .
</item 7>

<item 8>
Dead Sea
Highway 65 runs north - south on the Jordanian side from near Jordan 's northern tip down past the Dead Sea to the port of Aqaba . Human History Mount Sodom ,
Israel , showing the so - called `` Lot 's Wife '' pillar ( made of halite like the rest of the mountain ) A cargo boat on the Dead Sea as seen on the Madaba
Map , from the 6th century AD Biblical period Dwelling in caves near the Dead Sea is recorded in the Hebrew Bible as having taken place before the Israelites
came to Canaan , and extensively at the time of King David . Just northwest of the Dead Sea is Jericho . Somewhere , perhaps on the southeastern shore , would
be the cities mentioned in the Book of Genesis which were said to have been destroyed in the time of Abraham : Sodom and Gomorra ( Genesis 18 ) and the three
other `` Cities of the Plain '' , Admah , Zeboim and Zoar ( Deuteronomy 29 : 23 ) . Zoar escaped destruction when Abraham 's nephew Lot escaped to Zoar from
Sodom ( Genesis 19 : 21 - 22 ) .
</item 8>

<item 9>
Dead Sea
In Roman and Byzantine times , sugarcane , henna , and sycamore fig all made the lower Jordan valley wealthy . One of the most valuable products produced by
Jericho was the sap of the balsam tree , which could be made into perfume . By the 19th century , Jericho 's fertility had disappeared . Human settlement There
are several small communities near the Dead Sea . These include Ein Gedi , Neve Zohar and the Israeli settlements in the Megilot Regional Council : Kalya ,
Mitzpe Shalem and Avnat . There is a nature preserve at Ein Gedi , and several Dead Sea hotels are located on the southwest end at Ein Bokek near Neve Zohar .
Highway 90 runs north - south on the Israeli side for a total distance of 565 km ( 351 mi ) from Metula on the Lebanese border in the north to its southern
terminus at the Egyptian border near the Red Sea port of Eilat . Potash City is a small community on the Jordanian side of the Dead Sea , and others including
Suweima .
</item 9>

<item 10>
Dead Sea
There are no outlet streams . The Mujib River , biblical Arnon , is one the larger water sources of the Dead Sea other than the Jordan . The Wadi Mujib valley ,
420 m below the sea level in the southern of Jordan valley , is a biosphere reserve , with an area of 212 km ( 82 sq mi ) . Other more substantial sources are
Wadi Darajeh ( Arabic ) / Nahal Dragot ( Hebrew ) , and Nahal Arugot . Wadi Hasa ( biblical Zered ) is another wadi flowing into the Dead Sea . Rainfall is
scarcely 100 mm ( 4 in ) per year in the northern part of the Dead Sea and barely 50 mm ( 2 in ) in the southern part . The Dead Sea zone 's aridity is due to
the rainshadow effect of the Judaean Mountains . The highlands east of the Dead Sea receive more rainfall than the Dead Sea itself . To the west of the Dead Sea
, the Judaean mountains rise less steeply and are much lower than the mountains to the east .
</item 10>
=== END ITEM id=244937a027d78b8a ===

=== ITEM id=139723cd88d5613a ===
Gold sentences and facts:
Sentence 0: The Day the Music Died The memorial marking the crash site , 2003 Accident summary Date February 3 , 1959 ( 1959 - 02 - 03 ) Summary Spatial disorientation , loss of control in near - IMC conditions Site Near Clear Lake , Iowa , United States 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿ / ﻿ 43.220361 ° N 93.381417 ° W ﻿ / 43.220361 ; - 93.381417 Coordinates : 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿ / ﻿ 43.220361 ° N 93.381417 ° W ﻿ / 43.220361 ; - 93.381417 Passengers Crew Fatalities 4 ( all ) Survivors 0 Aircraft type Beechcraft Bonanza Operator Dwyer Flying Service , Mason City , Iowa Registration N3794N Flight origin Mason City Municipal Airport , Iowa Destination Hector Airport , North Dakota
  Fact 0.0: The Day the Music Died has a memorial marking the crash site, photographed in 2003.
  Fact 0.1: The accident date was February 3, 1959 (1959-02-03).
  Fact 0.2: The accident summary is spatial disorientation and loss of control in near-IMC conditions.
  Fact 0.3: The accident site is near Clear Lake, Iowa, United States.
  Fact 0.4: The accident site coordinates are 43°13′13.3″N 93°22′53.1″W.
  Fact 0.5: The accident site coordinates are 43.220361°N 93.381417°W.
  Fact 0.6: The accident had 4 fatalities, all of the people involved.
  Fact 0.7: The accident had 0 survivors.
  Fact 0.8: The aircraft type was a Beechcraft Bonanza.
  Fact 0.9: The operator was Dwyer Flying Service, of Mason City, Iowa.
  Fact 0.10: The aircraft registration was N3794N.
  Fact 0.11: The flight origin was Mason City Municipal Airport, Iowa.
  Fact 0.12: The flight destination was Hector Airport, North Dakota.
Sentence 1: At the time , Holly and his band , consisting of Waylon Jennings , Tommy Allsup , and Carl Bunch , were playing on the `` Winter Dance Party '' tour across the United States Midwest . Rising artists Valens and Richardson had joined the tour , as well . The long journeys between venues on board the cold , uncomfortable tour buses adversely affected the performers , with cases of flu and even frostbite . After stopping at Clear Lake to perform , and frustrated by such conditions , Holly chose to charter a plane to reach their next venue in Moorhead , Minnesota . Richardson , who had the flu , swapped places with Jennings , taking his seat on the plane , while Allsup lost his seat to Valens on a coin toss . Soon after takeoff , late at night and in poor , wintry weather conditions , the pilot lost control of the light aircraft , a Beechcraft Bonanza , which subsequently crashed into a cornfield . Everyone on board was killed . The event has since been mentioned in various songs and films . A number of monuments have been erected at the crash site and in Clear Lake , where an annual memorial concert is also held at the Surf Ballroom , the venue that hosted the artists ' last performance .
  Fact 1.0: At the time, Holly and his band were playing on the 'Winter Dance Party' tour across the United States Midwest.
  Fact 1.1: Holly's band consisted of Waylon Jennings, Tommy Allsup, and Carl Bunch.
  Fact 1.2: Rising artists Valens and Richardson had joined the tour.
  Fact 1.3: The long journeys between venues on board the cold, uncomfortable tour buses adversely affected the performers.
  Fact 1.4: The performers had cases of flu.
  Fact 1.5: The performers had cases of frostbite.
  Fact 1.6: After stopping at Clear Lake to perform, and frustrated by such conditions, Holly chose to charter a plane.
  Fact 1.7: Holly chartered the plane to reach the next venue in Moorhead, Minnesota.
  Fact 1.8: Richardson, who had the flu, swapped places with Jennings, taking Jennings's seat on the plane.
  Fact 1.9: Allsup lost his seat to Valens on a coin toss.
  Fact 1.10: Soon after takeoff, late at night and in poor, wintry weather conditions, the pilot lost control of the light aircraft.
  Fact 1.11: The light aircraft was a Beechcraft Bonanza.
  Fact 1.12: The aircraft subsequently crashed into a cornfield.
  Fact 1.13: Everyone on board was killed.
  Fact 1.14: The event has since been mentioned in various songs and films.
  Fact 1.15: A number of monuments have been erected at the crash site.
  Fact 1.16: A number of monuments have been erected in Clear Lake.
  Fact 1.17: An annual memorial concert is also held at the Surf Ballroom.
  Fact 1.18: The Surf Ballroom was the venue that hosted the artists' last performance.

Retrieved items:
<item 1>
The Day the Music Died
.. His Greatest Hits Volume 2 History of Ritchie Valens The Best of Ritchie Valens Songs `` Donna '' `` Bony Moronie '' `` La Bamba '' `` Bluebirds over the
Mountain '' `` We Belong Together '' Related Del - Fi Records Chicano rock Bob Keane The Day the Music Died ( 1959 plane crash ) La Bamba ( film ) ← 1958
Aviation accidents and incidents in 1959 ( 1959 ) 1960 → Jan 8 Southeast Airlines Flight 308 Jan 11 Lufthansa Flight 502 Feb 3 American Airlines Flight 320 Feb
3 Clear Lake crash ( `` The Day the Music Died '' ) Feb 17 Turkish Airlines Gatwick crash Apr 23 Air Charter Turkey crash May 12 Capital Airlines Flight 75 Jun
26 TWA Flight 891 Jun 30 Okinawa F - 100 crash Aug 15 American Airlines Flight 514 Aug 19 Transair Douglas Dakota accident Sep 24 TAI Flight 307 Sep 29 Braniff
Flight 542 Oct 30 Piedmont Airlines Flight 349 Nov 16 National Airlines Flight 967 Nov 21 Ariana Afghan Airlines Flight 202 Retrieved from `` https://en.
</item 1>

<item 2>
The Day the Music Died
Mason City , Iowa Registration N3794N Flight origin Mason City Municipal Airport , Iowa Destination Hector Airport , North Dakota On February 3 , 1959 , rock
and roll musicians Buddy Holly , Ritchie Valens , and J.P. `` The Big Bopper '' Richardson were killed in a plane crash near Clear Lake , Iowa , together with
pilot Roger Peterson . The event later became known as `` The Day the Music Died '' , after singer - songwriter Don McLean referred to it as such in his 1971
song `` American Pie '' . At the time , Holly and his band , consisting of Waylon Jennings , Tommy Allsup , and Carl Bunch , were playing on the `` Winter Dance
Party '' tour across the United States Midwest . Rising artists Valens and Richardson had joined the tour , as well . The long journeys between venues on board
the cold , uncomfortable tour buses adversely affected the performers , with cases of flu and even frostbite . After stopping at Clear Lake to perform , and
frustrated by such conditions , Holly chose to charter a plane to reach their next venue in Moorhead , Minnesota .
</item 2>

<item 3>
The Day the Music Died
The rest of the party would have picked him up in Moorhead , saving him the journey in the bus and leaving him time to get some rest . Flight arrangements (
edit ) A V - tailed Bonanza similar to N3794N , the accident aircraft Surf Ballroom manager Anderson called Hubert Jerry Dwyer ( 1930 - 2016 ) , owner of the
Dwyer Flying Service , a company in Mason City , Iowa , to charter the plane to fly to Hector Airport in Fargo , the closest one to Moorhead . Flight
arrangements were made with Roger Peterson , a 21 - year - old local pilot described as a `` young married man who built his life around flying '' . The flying
service charged a fee of $36 per passenger for the flight on the 1947 single - engined , V - tailed Beechcraft 35 Bonanza ( registration N3794N ) , which could
seat three passengers plus the pilot . A popular misconception , originating from Don McLean 's eponymous song about the crash , was that the plane was called
American Pie . In fact , no record exists of any name ever having been given to N3794N .
</item 3>

<item 4>
The Day the Music Died
The tail light was then observed gradually descending until it disappeared out of view . Around 1 : 00 am , when Peterson failed to make the expected radio
contact , repeated attempts to establish communication were made , at Dwyer 's request , by the radio operator , but they were all unsuccessful . Later that
morning , Dwyer , having heard no word from Peterson since his departure , took off in another airplane to retrace his planned route . Within minutes , at
around 9 : 35 am , he spotted the wreckage less than 6 mi ( 9.7 km ) northwest of the airport . The sheriff 's office , alerted by Dwyer , dispatched Deputy
Bill McGill , who drove to the crash site , a cornfield belonging to Albert Juhl . The Bonanza had impacted terrain at high speed , estimated to have been
around 170 mph ( 270 km / h ) , banked steeply to the right and in a nose - down attitude .
</item 4>

<item 5>
The Day the Music Died
In fact , no record exists of any name ever having been given to N3794N . The most widely accepted version of events was that Richardson had contracted flu
during the tour and asked Waylon Jennings for his seat on the plane . When Holly learned that Jennings was not going to fly , he said in jest : `` Well , I hope
your ol ' bus freezes up . '' Jennings responded : `` Well , I hope your ol ' plane crashes '' , a humorous but ill - fated response that haunted him for the
rest of his life . Ritchie Valens , who once had a fear of flying , asked Tommy Allsup for his seat on the plane . The two agreed to toss a coin to decide . Bob
Hale , a DJ with KRIB - AM , was working the concert that night and flipped the coin in the ballroom 's side - stage room shortly before the musicians departed
for the airport . Valens won the coin toss for the seat on the flight . In contradiction to the testimony of Allsup and Jennings , Dion has since said that
Holly approached him along with Valens and The Bopper to join the flight , not Holly 's bandmates .
</item 5>

<item 6>
The Day the Music Died
The right wing tip had struck the ground first , sending the aircraft cartwheeling across the frozen field for 540 feet ( 160 m ) , before coming to rest
against a wire fence at the edge of Juhl 's property . The bodies of Holly and Valens had been ejected from the torn fuselage and lay near the wreckage .
Richardson 's body had been thrown over the fence and into the cornfield of Juhl 's neighbor Oscar Moffett , while pilot Peterson 's body was entangled in the
plane 's wreckage . With the rest of the entourage en route to Minnesota , ballroom manager Carroll Anderson , who had driven the party to the airport and
witnessed the plane 's takeoff , had to identify the bodies of the musicians . County coroner Ralph Smiley certified that all four victims died instantly ,
citing the cause of death as `` gross trauma to brain '' for the three artists and `` brain damage '' for the pilot . Aftermath ( edit ) Holly 's pregnant wife
, María Elena , learned of his death from the reports on television .
</item 6>

<item 7>
The Day the Music Died
The musician 's son , Jay Perry , took the opportunity to have his father 's body re-examined to verify the original findings , and asked forensic
anthropologist William Bass to carry out the procedure . Among the rumors surrounding the accident that this second examination sought to verify was that an
accidental firearm discharge took place on board the aircraft and caused the crash , since two months after the event , a farmer had found at the crash site a .
22 ( 5.6 mm ) pistol known to have belonged to Buddy Holly . Another rumor had Richardson surviving the initial impact and crawling out of the aircraft in
search for help , prompted by the fact that his body was found farther from the wreckage than the other three . Bass and his team took several X-rays of
Richardson 's body and eventually concluded that the musician had indeed died instantly from extensive , nonsurvivable fractures to almost all of his bones ; no
traces of lead were found from any bullet . Coroner Smiley 's original report was , therefore , confirmed . In March 2015 , the National Transportation Safety
Board ( NTSB ) received a request to reopen the investigation into the accident . The request was made by L.J.
</item 7>

<item 8>
The Day the Music Died
The Day the Music Died - wikipedia The Day the Music Died Jump to : navigation , search This article is about the plane crash . For other uses , see The Day the
Music Died ( disambiguation ) . The Day the Music Died The memorial marking the crash site , 2003 Accident summary Date February 3 , 1959 ( 1959 - 02 - 03 )
Summary Spatial disorientation , loss of control in near - IMC conditions Site Near Clear Lake , Iowa , United States 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿
/ ﻿ 43.220361 ° N 93.381417 ° W ﻿ / 43.220361 ; - 93.381417 Coordinates : 43 ° 13 ′ 13.3 '' N 93 ° 22 ′ 53.1 '' W ﻿ / ﻿ 43.220361 ° N 93.381417 ° W ﻿ /
43.220361 ; - 93.381417 Passengers Crew Fatalities 4 ( all ) Survivors 0 Aircraft type Beechcraft Bonanza Operator Dwyer Flying Service ,
</item 8>

<item 9>
The Day the Music Died
Richardson , who had the flu , swapped places with Jennings , taking his seat on the plane , while Allsup lost his seat to Valens on a coin toss . Soon after
takeoff , late at night and in poor , wintry weather conditions , the pilot lost control of the light aircraft , a Beechcraft Bonanza , which subsequently
crashed into a cornfield . Everyone on board was killed . The event has since been mentioned in various songs and films . A number of monuments have been
erected at the crash site and in Clear Lake , where an annual memorial concert is also held at the Surf Ballroom , the venue that hosted the artists ' last
performance . Contents ( hide ) 1 Background 2 Flight arrangements 3 Take - off and crash 4 Aftermath 5 Official investigation 6 Subsequent investigations 7
Legacy 7.1 Notification of victims ' families 7.2 Memorials 7.2. 1 Films 7.2. 2 Memorial concerts 7.2. 3 Monuments 7.2. 4 Roads 7.2.
</item 9>

<item 10>
The Day the Music Died
Don McLean later addressed the accident in his song `` American Pie '' ( 1971 ) , dubbing it `` the Day the Music Died '' , which for McLean symbolized the ``
loss of innocence '' of the early rock - and - roll generation . See also ( edit ) Aviation portal Iowa portal Rock and Roll portal Patsy Cline plane crash
Stevie Ray Vaughan helicopter crash Bill Graham helicopter crash Randy Rhoads plane crash 1969 Newton Cessna 172 crash Lynyrd Skynyrd plane crash References (
edit ) ^ Jump up to : Suddath , Claire ( February 3 , 2009 ) . `` The Day the Music Died '' . Time . Retrieved April 29 , 2015 . Jump up ^ Everitt 2004 , p. 10
. ^ Jump up to : http://www.startribune.com/buddy-holly-the-tour-from-hell/38282249/ ^ Jump up to : Everitt 2004 , p. 13 . Jump up ^ `` Connection to Buddy
Holly Death '' . WeGoNews.com . Retrieved February 3 , 2011 .
</item 10>
=== END ITEM id=139723cd88d5613a ===

=== ITEM id=ed8359315b630f2a ===
Gold sentences and facts:
Sentence 0: ( hide ) Climate data for The South Pole Month Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec Year Record high ° C ( ° F ) − 14.4 ( 6.1 ) − 20.6 ( − 5.1 ) − 26.7 ( − 16.1 ) − 27.8 ( − 18 ) − 25.1 ( − 13.2 ) − 28.8 ( − 19.8 ) − 33.9 ( − 29 ) − 32.8 ( − 27 ) − 29.3 ( − 20.7 ) − 25.1 ( − 13.2 ) − 18.9 ( − 2 ) − 12.3 ( 9.9 ) − 12.3 ( 9.9 ) Average high ° C ( ° F ) − 26.0 ( − 14.8 ) − 37.9 ( − 36.2 ) − 49.6 ( − 57.3 ) − 53.0 ( − 63.4 ) − 53.6 ( − 64.5 ) − 54.5 ( − 66.1 ) − 55.2 ( − 67.4 ) − 54.9 ( − 66.8 ) − 54.4 ( − 65.9 ) − 48.4 ( − 55.1 ) − 36.2 ( − 33.2 ) − 26.3 ( − 15.3 ) − 45.8 ( − 50.4 ) Daily mean ° C ( ° F ) − 28.4 ( − 19.1 ) − 40.9 ( − 41.6 ) − 53.7 ( − 64.7 ) − 57.8 ( − 72 ) − 58.0 ( − 72.4 ) − 58.9 ( − 74 ) − 59.8 ( − 75.6 ) − 59.7 ( − 75.5 ) − 59.1 ( − 74.4 ) − 51.6 ( − 60.9 ) − 38.2 ( − 36.8 ) − 28.0 ( − 18.4 ) − 49.5 ( − 57.1 ) Average low ° C ( ° F ) − 29.6 ( − 21.3 ) − 43.1 ( − 45.6 ) − 56.8 ( − 70.2 ) − 60.9 ( − 77.6 ) − 61.5 ( − 78.7 ) − 62.8 ( − 81 ) − 63.4 ( − 82.1 ) − 63.2 ( − 81.8 ) − 61.7 ( − 79.1 ) − 54.3 ( − 65.7 ) − 40.1 ( − 40.2 ) − 29.1 ( − 20.4 ) − 52.2 ( − 62 ) Record low ° C ( ° F ) − 41.1 ( − 42 ) − 58.9 ( − 74 ) − 71.1 ( − 96 ) − 75.0 ( − 103 ) − 78.3 ( − 108.9 ) − 82.8 ( − 117 ) − 80.6 ( − 113.1 ) − 79.3 ( − 110.7 ) − 79.4 ( − 110.9 ) − 72.0 ( − 97.6 ) − 55.0 ( − 67 ) − 41.1 ( − 42 ) − 82.8 ( − 117 ) Average precipitation mm ( inches ) 0.3 ( 0.012 ) 0.6 ( 0.024 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) Trace Trace 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.3 ( 0.012 ) 2.3 ( 0.091 ) Average precipitation days ( ≥ 0.1 mm ) 0.2 0.3 0.2 0.0 0.2 0.1 0.0 0.0 0.1 0.1 0.1 0.3 1.6 Average snowy days 22.0 19.6 13.6 11.4 17.2 17.3 18.2 17.5 11.7 16.7 16.9 20.6 203.0 Mean monthly sunshine hours 406.1 497.2 195.3 0.0 0.0 0.0 0.0 0.0 34.1 390.6 558.0 616.9 2,698.2 Mean daily sunshine hours 13.1 17.6 6.3 0.0 0.0 0.0 0.0 0.0 1.1 12.6 18.6 19.9 7.4 Source # 1 : Pogoda.ru.net ( temperatures , 1981 -- 2010 , extremes 1957 -- present ) Source # 2 : Deutscher Wetterdienst ( precipitation 1957 -- 1988 and sun 1978 -- 1993 ) , NOAA ( snowy days data , 1961 -- 1988 )
  Fact 0.0: The climate data table is for The South Pole.
  Fact 0.1: The record high temperatures at The South Pole in degrees Celsius (Fahrenheit) are -14.4 (6.1) in January, -20.6 (-5.1) in February, -26.7 (-16.1) in March, -27.8 (-18) in April, -25.1 (-13.2) in May, -28.8 (-19.8) in June, -33.9 (-29) in July, -32.8 (-27) in August, -29.3 (-20.7) in September, -25.1 (-13.2) in October, -18.9 (-2) in November and -12.3 (9.9) in December.
  Fact 0.2: The record high temperature for the year at The South Pole is -12.3 (9.9) degrees Celsius (Fahrenheit).
  Fact 0.3: The average high temperatures at The South Pole in degrees Celsius (Fahrenheit) are -26.0 (-14.8) in January, -37.9 (-36.2) in February, -49.6 (-57.3) in March, -53.0 (-63.4) in April, -53.6 (-64.5) in May, -54.5 (-66.1) in June, -55.2 (-67.4) in July, -54.9 (-66.8) in August, -54.4 (-65.9) in September, -48.4 (-55.1) in October, -36.2 (-33.2) in November and -26.3 (-15.3) in December.
  Fact 0.4: The average high temperature for the year at The South Pole is -45.8 (-50.4) degrees Celsius (Fahrenheit).
  Fact 0.5: The daily mean temperatures at The South Pole in degrees Celsius (Fahrenheit) are -28.4 (-19.1) in January, -40.9 (-41.6) in February, -53.7 (-64.7) in March, -57.8 (-72) in April, -58.0 (-72.4) in May, -58.9 (-74) in June, -59.8 (-75.6) in July, -59.7 (-75.5) in August, -59.1 (-74.4) in September, -51.6 (-60.9) in October, -38.2 (-36.8) in November and -28.0 (-18.4) in December.
  Fact 0.6: The daily mean temperature for the year at The South Pole is -49.5 (-57.1) degrees Celsius (Fahrenheit).
  Fact 0.7: The average low temperatures at The South Pole in degrees Celsius (Fahrenheit) are -29.6 (-21.3) in January, -43.1 (-45.6) in February, -56.8 (-70.2) in March, -60.9 (-77.6) in April, -61.5 (-78.7) in May, -62.8 (-81) in June, -63.4 (-82.1) in July, -63.2 (-81.8) in August, -61.7 (-79.1) in September, -54.3 (-65.7) in October, -40.1 (-40.2) in November and -29.1 (-20.4) in December.
  Fact 0.8: The average low temperature for the year at The South Pole is -52.2 (-62) degrees Celsius (Fahrenheit).
  Fact 0.9: The record low temperatures at The South Pole in degrees Celsius (Fahrenheit) are -41.1 (-42) in January, -58.9 (-74) in February, -71.1 (-96) in March, -75.0 (-103) in April, -78.3 (-108.9) in May, -82.8 (-117) in June, -80.6 (-113.1) in July, -79.3 (-110.7) in August, -79.4 (-110.9) in September, -72.0 (-97.6) in October, -55.0 (-67) in November and -41.1 (-42) in December.
  Fact 0.10: The record low temperature for the year at The South Pole is -82.8 (-117) degrees Celsius (Fahrenheit).
  Fact 0.11: The average precipitation at The South Pole in millimeters (inches) is 0.3 (0.012) in January, 0.6 (0.024) in February, 0.2 (0.008) in March, 0.1 (0.004) in April, 0.2 (0.008) in May, 0.1 (0.004) in June, Trace in July, Trace in August, 0.1 (0.004) in September, 0.1 (0.004) in October, 0.1 (0.004) in November and 0.3 (0.012) in December.
  Fact 0.12: The average precipitation for the year at The South Pole is 2.3 (0.091) millimeters (inches).
  Fact 0.13: The average number of days with precipitation of at least 0.1 mm at The South Pole is 0.2 in January, 0.3 in February, 0.2 in March, 0.0 in April, 0.2 in May, 0.1 in June, 0.0 in July, 0.0 in August, 0.1 in September, 0.1 in October, 0.1 in November and 0.3 in December.
  Fact 0.14: The average number of days with precipitation of at least 0.1 mm at The South Pole is 1.6 for the year.
  Fact 0.15: The average number of snowy days at The South Pole is 22.0 in January, 19.6 in February, 13.6 in March, 11.4 in April, 17.2 in May, 17.3 in June, 18.2 in July, 17.5 in August, 11.7 in September, 16.7 in October, 16.9 in November and 20.6 in December.
  Fact 0.16: The average number of snowy days at The South Pole is 203.0 for the year.
  Fact 0.17: The mean monthly sunshine hours at The South Pole are 406.1 in January, 497.2 in February, 195.3 in March, 0.0 in April, 0.0 in May, 0.0 in June, 0.0 in July, 0.0 in August, 34.1 in September, 390.6 in October, 558.0 in November and 616.9 in December.
  Fact 0.18: The mean monthly sunshine hours at The South Pole total 2,698.2 for the year.
  Fact 0.19: The mean daily sunshine hours at The South Pole are 13.1 in January, 17.6 in February, 6.3 in March, 0.0 in April, 0.0 in May, 0.0 in June, 0.0 in July, 0.0 in August, 1.1 in September, 12.6 in October, 18.6 in November and 19.9 in December.
  Fact 0.20: The mean daily sunshine hours at The South Pole are 7.4 for the year.
  Fact 0.21: Source 1 for the data is Pogoda.ru.net, for temperatures from 1981 to 2010 and extremes from 1957 to present.
  Fact 0.22: Source 2 for the data is Deutscher Wetterdienst, for precipitation from 1957 to 1988 and sun from 1978 to 1993.
  Fact 0.23: Source 2 also includes NOAA, for snowy days data from 1961 to 1988.

Retrieved items:
<item 1>
South Pole
0 0.0 1.1 12.6 18.6 19.9 7.4 Source # 1 : Pogoda.ru.net ( temperatures , 1981 -- 2010 , extremes 1957 -- present ) Source # 2 : Deutscher Wetterdienst (
precipitation 1957 -- 1988 and sun 1978 -- 1993 ) , NOAA ( snowy days data , 1961 -- 1988 ) Time ( edit ) In most places on Earth , local time is determined by
longitude , such that the time of day is more - or-less synchronised to the position of the sun in the sky ( for example , at midday the sun is roughly at its
highest ) . This line of reasoning fails at the South Pole , where the sun rises and sets only once per year , and all lines of longitude , and hence all time
zones , converge . There is no a priori reason for placing the South Pole in any particular time zone , but as a matter of practical convenience the Amundsen --
Scott South Pole Station keeps New Zealand Time ( UTC + 12 ) .
</item 1>

<item 2>
North Pole
However , a freak storm caused the temperature to reach 0.7 ° C ( 33 ° F ) for a time at a World Meteorological Organization buoy , located at 87.45 ° N , on
December 30 , 2015 . It was estimated that the temperature at the North Pole was between 30 and 35 ° F ( − 1 and 2 ° C ) during the storm . Summer temperatures
( June , July , and August ) average around the freezing point ( 0 ° C ( 32 ° F ) ) . The highest temperature yet recorded is 13 ° C ( 55 ° F ) , much warmer
than the South Pole 's record high of only − 12.3 ° C ( 9.9 ° F ) . A similar spike in temperatures occurred on November 15 , 2016 when temperatures hit
freezing . Yet again , February of 2018 featured a storm so powerful that temperatures at Cape Morris Jesup , the world northermost weather station in Greenland
, reached 6.1 ° C ( 43 ° F ) and spent 24 straight hours above freezing .
</item 2>

<item 3>
South Pole
2 ) − 49.6 ( − 57.3 ) − 53.0 ( − 63.4 ) − 53.6 ( − 64.5 ) − 54.5 ( − 66.1 ) − 55.2 ( − 67.4 ) − 54.9 ( − 66.8 ) − 54.4 ( − 65.9 ) − 48.4 ( − 55.1 ) − 36.2 ( −
33.2 ) − 26.3 ( − 15.3 ) − 45.8 ( − 50.4 ) Daily mean ° C ( ° F ) − 28.4 ( − 19.1 ) − 40.9 ( − 41.6 ) − 53.7 ( − 64.7 ) − 57.8 ( − 72 ) − 58.0 ( − 72.4 ) − 58.9
( − 74 ) − 59.8 ( − 75.6 ) − 59.7 ( − 75.5 ) − 59.1 ( − 74.4 ) − 51.
</item 3>

<item 4>
South Pole
1 ( − 42 ) − 58.9 ( − 74 ) − 71.1 ( − 96 ) − 75.0 ( − 103 ) − 78.3 ( − 108.9 ) − 82.8 ( − 117 ) − 80.6 ( − 113.1 ) − 79.3 ( − 110.7 ) − 79.4 ( − 110.9 ) − 72.0
( − 97.6 ) − 55.0 ( − 67 ) − 41.1 ( − 42 ) − 82.8 ( − 117 ) Average precipitation mm ( inches ) 0.3 ( 0.012 ) 0.6 ( 0.024 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) 0.2 (
0.008 ) 0.1 ( 0.004 ) Trace Trace 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.3 ( 0.012 ) 2.3 ( 0.
</item 4>

<item 5>
South Pole
More recent buildings are raised on stilts so that the snow does not build up against the sides of them . ( hide ) Climate data for The South Pole Month Jan Feb
Mar Apr May Jun Jul Aug Sep Oct Nov Dec Year Record high ° C ( ° F ) − 14.4 ( 6.1 ) − 20.6 ( − 5.1 ) − 26.7 ( − 16.1 ) − 27.8 ( − 18 ) − 25.1 ( − 13.2 ) − 28.8
( − 19.8 ) − 33.9 ( − 29 ) − 32.8 ( − 27 ) − 29.3 ( − 20.7 ) − 25.1 ( − 13.2 ) − 18.9 ( − 2 ) − 12.3 ( 9.9 ) − 12.3 ( 9.9 ) Average high ° C ( ° F ) − 26.0 ( −
14.8 ) − 37.9 ( − 36.2 ) − 49.6 ( − 57.3 ) − 53.
</item 5>

<item 6>
South Pole
However , it is always low in the sky , reaching a maximum of 23.5 ° in December . Much of the sunlight that does reach the surface is reflected by the white
snow . This lack of warmth from the sun , combined with the high altitude ( about 2,800 metres ( 9,200 ft ) ) , means that the South Pole has one of the coldest
climates on Earth ( though it is not quite the coldest ; that record goes to the region in the vicinity of the Vostok Station , also in Antarctica , which lies
at a higher elevation ) . Temperatures at the South Pole are much lower than at the North Pole , primarily because the South Pole is located at altitude in the
middle of a continental land mass , while the North Pole is at sea level in the middle of an ocean , which acts as a reservoir of heat . The South Pole is at an
altitude of 9,300 feet ( 2,800 m ) but feels like 11,000 feet ( 3,400 m ). . Centrifugal force from the spin of the planet pulls the atmosphere toward the
equator .
</item 6>

<item 7>
South Pole
The geographic coordinates of the South Pole are usually given simply as 90 ° S , since its longitude is geometrically undefined and irrelevant . When a
longitude is desired , it may be given as 0 ° . At the South Pole , all directions face north . For this reason , directions at the Pole are given relative to
`` grid north '' , which points northwards along the prime meridian . Along tight latitude circles , clockwise is east , and counterclockwise is west , opposite
to the North Pole . The Geographic South Pole is located on the continent of Antarctica ( although this has not been the case for all of Earth 's history
because of continental drift ) . It sits atop a featureless , barren , windswept and icy plateau at an altitude of 2,835 metres ( 9,301 ft ) above sea level ,
and is located about 1,300 km ( 800 mi ) from the nearest open sea at Bay of Whales . The ice is estimated to be about 2,700 metres ( 9,000 ft ) thick at the
Pole , so the land surface under the ice sheet is actually near sea level .
</item 7>

<item 8>
South Pole
5 ) − 59.1 ( − 74.4 ) − 51.6 ( − 60.9 ) − 38.2 ( − 36.8 ) − 28.0 ( − 18.4 ) − 49.5 ( − 57.1 ) Average low ° C ( ° F ) − 29.6 ( − 21.3 ) − 43.1 ( − 45.6 ) − 56.8
( − 70.2 ) − 60.9 ( − 77.6 ) − 61.5 ( − 78.7 ) − 62.8 ( − 81 ) − 63.4 ( − 82.1 ) − 63.2 ( − 81.8 ) − 61.7 ( − 79.1 ) − 54.3 ( − 65.7 ) − 40.1 ( − 40.2 ) − 29.1
( − 20.4 ) − 52.2 ( − 62 ) Record low ° C ( ° F ) − 41.1 ( − 42 ) − 58.
</item 8>

<item 9>
South Pole
Centrifugal force from the spin of the planet pulls the atmosphere toward the equator . The South Pole is colder than the North Pole because of the elevation
difference . The North Pole is a few feet from sea level . In midsummer , as the sun reaches its maximum elevation of about 23.5 degrees , high temperatures at
the South Pole in January average at − 25.9 ° C ( − 15 ° F ) . As the six - month `` day '' wears on and the sun gets lower , temperatures drop as well : they
reach − 45 ° C ( − 49 ° F ) around sunset ( late March ) and sunrise ( late September ) . In midwinter , the average temperature remains steady at around − 58 °
C ( − 72 ° F ) .
</item 9>

<item 10>
South Pole
The highest temperature ever recorded at the Amundsen -- Scott South Pole Station was − 12.3 ° C ( 9.9 ° F ) on Christmas Day , 2011 , and the lowest was − 82.8
° C ( − 117.0 ° F ) on June 23 , 1982 ( for comparison , the lowest temperature directly recorded anywhere on earth was − 89.2 ° C ( − 128.6 ° F ) at Vostok
Station on July 21 , 1983 , though − 93.2 ° C ( − 135.8 ° F ) was measured indirectly by satellite in East Antarctica between Dome A and Dome F in August 2010 )
. The South Pole has a desert climate , receiving very little precipitation . Air humidity is near zero . However , high winds can cause the blowing of snowfall
, and the accumulation of snow amounts to about 20 cm ( 8 in ) per year . The former dome seen in pictures of the Amundsen -- Scott station is partially buried
due to snow storms , and the entrance to the dome had to be regularly bulldozed to uncover it .
</item 10>
=== END ITEM id=ed8359315b630f2a ===

=== ITEM id=9e56e44b8561b477 ===
Gold sentences and facts:
Sentence 0: Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa . Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order , an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star Wars : The Last Jedi ( 2017 ) .
  Fact 0.0: Kylo Ren is a fictional character in the Star Wars franchise.
  Fact 0.1: Kylo Ren was introduced in the 2015 film Star Wars: The Force Awakens.
  Fact 0.2: Kylo Ren is portrayed by Adam Driver.
  Fact 0.3: Kylo Ren is the chosen name of Ben Solo.
  Fact 0.4: Ben Solo is the son of original Star Wars trilogy characters Han Solo and Leia Organa.
  Fact 0.5: Kylo Ren was trained by his uncle Luke Skywalker as a Jedi.
  Fact 0.6: Kylo Ren has been seduced to the dark side of the Force by Supreme Leader Snoke.
  Fact 0.7: Kylo Ren aspires to be as powerful as his grandfather, Darth Vader.
  Fact 0.8: Kylo Ren is the master of the Knights of Ren.
  Fact 0.9: Kylo Ren is a commander of the First Order.
  Fact 0.10: Kylo Ren later became the supreme leader of the First Order.
  Fact 0.11: The First Order is an organization spawned from the fallen Galactic Empire.
  Fact 0.12: Kylo Ren is featured in The Force Awakens media and merchandising.
  Fact 0.13: Kylo Ren appears in the film's sequel, Star Wars: The Last Jedi.
  Fact 0.14: Star Wars: The Last Jedi was released in 2017.
Sentence 1: Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous . Melissa Leon of The Daily Beast describes Ren 's use of the Force as `` formidable '' , citing his ability to stop a blaster shot mid-air , immobilize victims and probe their minds against their will .
  Fact 1.0: Abrams told Empire in August 2015 that Kylo Ren is not a Sith.
  Fact 1.1: Abrams told Empire in August 2015 that Kylo Ren works under Supreme Leader Snoke.
  Fact 1.2: Abrams told Empire in August 2015 that Supreme Leader Snoke is a powerful figure on the Dark Side of the Force.
  Fact 1.3: Abrams had previously told Entertainment Weekly that the character came to the name Kylo Ren when he joined a group called the Knights of Ren.
  Fact 1.4: Robbie Collin of The Telegraph described Ren as a hot-headed, radicalised Dark Side jihadi.
  Fact 1.5: Robbie Collin of The Telegraph described Ren's red lightsaber as spluttering and crackling as violently as his temper.
  Fact 1.6: Abrams noted that the lightsaber is something Ren built himself.
  Fact 1.7: Abrams noted that the lightsaber is as dangerous, fierce and ragged as the character.
  Fact 1.8: The Telegraph explains that Ren's wild and erratic temper and angsty instability make him dangerous.
  Fact 1.9: Melissa Leon of The Daily Beast describes Ren's use of the Force as formidable.
  Fact 1.10: Melissa Leon cites Ren's ability to stop a blaster shot mid-air.
  Fact 1.11: Melissa Leon cites Ren's ability to immobilize victims.
  Fact 1.12: Melissa Leon cites Ren's ability to probe victims' minds against their will.

Retrieved items:
<item 1>
Kylo Ren
Novels ( edit ) Kylo Ren is a point of view character in the 2015 novelization of The Force Awakens by Alan Dean Foster . In the 2017 Chuck Wendig novel Star
Wars : Aftermath : Empire 's End , Ben Solo is stated as having been born on the planet Chandrila on the same day as a peace treaty is signed between the
remnants of the Empire and the New Galactic Republic . Kylo Ren is additionally mentioned in the 2017 novel Star Wars : Phasma . Reception ( edit ) Adam Driver
received positive reviews for his performance as Kylo Ren . Both the character of Kylo Ren and Driver 's performance have been positively received . Many
complimented Ren 's conflicted nature and depth , as well as his costume design , and noted there were many places the character could be taken in future
installments . Peter Bradshaw of The Guardian praised the character and the actor alike , saying , `` He is gorgeously cruel , spiteful and capricious -- and
unlike the Vader of old , he is given to petulant temper tantrums , with his lightsaber drawn . ''
</item 1>

<item 2>
Kylo Ren
Ren 's character arc shares similarities with that of the Star Wars expanded universe character Jacen Solo , the son of Han Solo and Princess Leia who threatens
the galaxy as a fallen Jedi . Additionally , critics have noted a resemblance between Ren 's character design and that of Revan , the protagonist of Knights of
the Old Republic . Character ( edit ) Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful
figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a
group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber
splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce
and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous .
</item 2>

<item 3>
Kylo Ren
Abrams and Michael Arndt , with no direct participation from Star Wars creator George Lucas . According to director Abrams , the First Order is inspired by
ODESSA , a theoretical network of SS officers who fled to Argentina following World War II . Driver 's casting in the film in an unnamed role was first
announced on April 29 , 2014 . Kylo Ren was first seen , but still not named , in the 88 - second The Force Awakens teaser trailer released by Lucasfilm on
November 28 , 2014 , wielding a jagged red lightsaber with a crossguard . The name Kylo Ren was revealed by Entertainment Weekly in a Lucasfilm - designed Topps
- style trading card mockup in December 2014 . A May 2015 Vanity Fair photo shoot by Annie Leibovitz confirmed that Driver would be portraying Kylo . In an
interview with Time , The Force Awakens costume designer Michael Kaplan said that Abrams requested that Ren 's mask be designed to be memorable for a child .
After numerous attempts to gain Abrams ' approval , the final design was selected .
</item 3>

<item 4>
Kylo Ren
Ren soon learns that the pilot had entrusted his astromech droid , BB - 8 , with the map . Poe flees with the help of rogue stormtrooper Finn , who later finds
BB - 8 , and the scavenger Rey . Finn , Rey , and BB - 8 escape Jakku in the Millennium Falcon , and are soon intercepted by the ship 's former owner , Han Solo
, and his co-pilot Chewbacca . It is revealed that Ren is the son of Han and Leia , and was once one of Luke 's Jedi pupils . He was drawn to the dark side of
the Force by the First Order 's Supreme Leader Snoke . He then helped to destroy his uncle Luke 's new Jedi Academy . However , Ren still feels the pull of the
light side of the Force and seeks the strength to overcome this weakness from his grandfather Darth Vader , whose burnt helmet is in Ren 's possession .
Arriving at the Maz Kanata castle , Ren captures Rey , who he senses has seen the map . While interrogating her with the Force , he realizes that she is strong
with the Force herself , though unaware of it .
</item 4>

<item 5>
Kylo Ren
Abrams Michael Arndt Portrayed by Adam Driver Voiced by Adam Driver ( Disney Infinity 3.0 , Lego Star Wars : The Force Awakens , Lego Star Wars : The Resistance
Rises ) Matthew Wood ( Star Wars Battlefront II ) Roger Craig Smith ( Star Wars Battlefront II ) Information Full name Ben Solo Species Human Gender Male
Occupation Supreme Leader of the First Order Master of the Knights of Ren Former Jedi Padawan Affiliation First Order Knights of Ren New Jedi Order ( formerly )
Family Han Solo ( father ) Leia Organa ( mother ) Relatives Luke Skywalker ( maternal uncle ) Anakin Skywalker ( maternal grandfather ) Padmé Amidala ( maternal
grandmother ) Bail Organa ( adopted maternal grandfather ) Breha Organa ( adopted maternal grandmother ) Shmi Skywalker ( maternal great - grandmother )
Homeworld Chandrila Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed
by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa .
</item 5>

<item 6>
Kylo Ren
Kylo Ren - wikipedia Kylo Ren Jump to : navigation , search `` Kylo '' redirects here . For the web browser , see Kylo ( web browser ) . Kylo Ren Ben Solo Star
Wars character Kylo Ren 's costume displayed on a mannequin First appearance The Force Awakens ( 2015 ) Created by Lawrence Kasdan J.J.
</item 6>

<item 7>
Kylo Ren
Before the duel is finished , they are separated by a seismic fissure created by the collapsing base . Rey and the others escape as Snoke orders General Hux to
evacuate the base and bring Ren to him to complete his training . The last Jedi ( 2017 ) ( edit ) Main article : Star Wars : The Last Jedi Ren 's inner conflict
continues into The Last Jedi , specifically through his conversations with Rey , with whom he connects through the Force . Rey believes that his conflict will
ultimately bring him back to the light side . Ren captures her and brings her before Snoke , who orders Ren to kill her . However , Ren uses the Force to ignite
a lightsaber at Snoke 's side and cut him in half , killing him . Ren beseeches Rey to join him in creating a new order , separate from the legacies created by
Snoke and Luke . When Rey refuses , Ren declares himself Supreme Leader of the First Order , and orders his forces to attack the Resistance base on Crait .
</item 7>

<item 8>
Kylo Ren
Jump up ^ Travers , Peter ( December 16 , 2015 ) . `` Star Wars : The Force Awakens '' . Rolling Stone . Retrieved December 21 , 2015 . Jump up ^ Buchanan ,
Kyle ( November 11 , 2015 ) . `` Let 's Discuss the Ending of Star Wars : The Force Awakens '' . Vulture.com . Retrieved December 19 , 2015 . Jump up ^ Cohen ,
David S. ( June 23 , 2016 ) . `` The Force Awakens Rings Up Eight Saturn Awards '' . Variety . Retrieved June 23 , 2016 . Jump up ^ Agar , Chris ( January 22 ,
2018 ) . `` Kylo Ren Voted Seventh - Greatest Movie Villain '' . Screen Rant . Retrieved January 24 , 2018 . External links ( edit ) Wikimedia Commons has media
related to Kylo Ren . Kylo Ren in the StarWars.com Databank Kylo Ren on Wookieepedia ,
</item 8>

<item 9>
Kylo Ren
You normally think of order , and structure , and full commitment and no hesitation ... he 's just a little bit more unpolished . It 's in his costume , in his
lightsaber -- how you kind of get the sense that it could just not work at any moment ; that it could just blow up . That 's kind of like a big metaphor for him
. Appearances ( edit ) Model of Kylo 's cross-bladed lightsaber on display at Star Wars Launch Bay at Disney 's Hollywood Studios ( 2015 ) . Feature films (
edit ) The Force Awakens ( 2015 ) ( edit ) Main article : Star Wars : The Force Awakens Kylo Ren first appears in The Force Awakens as the commander of the
First Order , a sinister regime that has risen from the remains of the old Galactic Empire . After arriving at Jakku to retrieve a map containing the
coordinates for the location of Luke Skywalker , Ren kills an old vicar known as Lor San Tekka and captures Resistance pilot Poe Dameron , who has also been
sent to recover the map by General Leia Organa .
</item 9>

<item 10>
Kylo Ren
Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful
as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order ,
an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star
Wars : The Last Jedi ( 2017 ) . Contents ( hide ) 1 Concept and creation 2 Character 3 Appearances 3.1 Feature films 3.1. 1 The Force Awakens ( 2015 ) 3.1. 2
The Last Jedi ( 2017 ) 3.2 Related works and promotional material 3.3 Novels 4 Reception 5 Family tree 6 References 7 External links Concept and creation ( edit
) After Lucasfilm was acquired by The Walt Disney Company in 2012 , the script for The Force Awakens was written by Lawrence Kasdan , J.J.
</item 10>
=== END ITEM id=9e56e44b8561b477 ===

=== ITEM id=2edab24496c1a9f7 ===
Gold sentences and facts:
Sentence 0: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid touching metal .
  Fact 0.0: Take action immediately upon hearing thunder.
  Fact 0.1: Anyone close enough to the storm to hear thunder can be struck by lightning.
  Fact 0.2: Avoid electrical appliances, including corded telephones.
  Fact 0.3: Cordless and wireless telephones are safe to use during a thunderstorm.
  Fact 0.4: Close and stay away from windows and doors.
  Fact 0.5: Glass can become a serious hazard in high wind.
  Fact 0.6: Do not bathe or shower.
  Fact 0.7: Plumbing conducts electricity.
  Fact 0.8: If driving, safely exit the roadway.
  Fact 0.9: If driving, turn on hazard lights.
  Fact 0.10: If driving, park.
  Fact 0.11: Remain in the vehicle and avoid touching metal.
Sentence 1: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning .
  Fact 1.0: Take action immediately upon hearing thunder.
  Fact 1.1: Anyone close enough to the storm to hear thunder can be struck by lightning.

Retrieved items:
<item 1>
Thunderstorm
The National Weather Service ( NWS ) in the United States recommends several precautions that people should take if thunderstorms are likely to occur : Know the
names of local counties , cities , and towns , as these are how warnings are described . Monitor forecasts and weather conditions and know whether thunderstorms
are likely in the area . Be alert for natural signs of an approaching storm . Cancel or reschedule outdoor events ( to avoid being caught outdoors when a storm
hits ) . Take action early so you have time to get to a safe place . Get inside a substantial building or hard - topped metal vehicle before threatening weather
arrives . If you hear thunder , get to the safe place immediately . Avoid open areas like hilltops , fields , and beaches , and do n't be or be near the tallest
objects in an area when thunderstorms are occurring . Do n't shelter under tall or isolated trees during thunderstorms . If in the woods , put as much distance
between you and any trees during thunderstorms . If in a group , spread out so that you increase the chances for survivors who could come to the aid of any
victims from a lightning strike .
</item 1>

<item 2>
Thunderstorm
Lightningsafety.noaa.gov. 2014 - 06 - 30 . Retrieved 2014 - 08 - 20 . Jump up ^ `` NWS JetStream -- Lightning Frequently Asked Questions '' . Srh.noaa.gov. 2014
- 06 - 28 . Retrieved 2014 - 08 - 20 . Jump up ^ `` You 're not safer crouching : Six things you did n't know about lightning '' . LA Times . Retrieved 2014 -
08 - 20 . Jump up ^ National Geographic Almanac of Geography , ISBN 0 - 7922 - 3877 - X , page 75 . Jump up ^ `` How many thunderstorms occur each year ? '' .
Thunderstorms . Sky Fire Productions . Archived from the original on 2007 - 07 - 11 . Retrieved 2006 - 08 - 23 . Jump up ^ National Weather Service JetStream (
2008 - 10 - 08 ) . `` Tropical Cyclone Hazards '' . National Weather Service Southern Region Headquarters .
</item 2>

<item 3>
Thunderstorm
A similar event occurred in Boscastle , England , on 16 August 2004 . Hazards ( edit ) Each year , many people are killed or seriously injured by severe
thunderstorms despite the advance warning . While severe thunderstorms are most common in the spring and summer , they can occur at just about any time of the
year . Cloud - to - ground lightning ( edit ) See also : Lightning strike and Wildfire A return stroke , cloud - to - ground lightning strike during a
thunderstorm . Cloud - to - ground lightning frequently occurs within the phenomena of thunderstorms and have numerous hazards towards landscapes and
populations . One of the more significant hazards lightning can pose is the wildfires they are capable of igniting . Under a regime of low precipitation ( LP )
thunderstorms , where little precipitation is present , rainfall can not prevent fires from starting when vegetation is dry as lightning produces a concentrated
amount of extreme heat . Direct damage caused by lightning strikes occurs on occasion . In areas with a high frequency for cloud - to - ground lightning , like
Florida , lightning causes several fatalities per year , most commonly to people working outside . Acid rain is also a frequent risk produced by lightning .
</item 3>

<item 4>
Thunderstorm
As the rising air reaches its dew point temperature , water vapor condenses into water droplets or ice , reducing pressure locally within the thunderstorm cell
. Any precipitation falls the long distance through the clouds towards the Earth 's surface . As the droplets fall , they collide with other droplets and become
larger . The falling droplets create a downdraft as it pulls cold air with it , and this cold air spreads out at the Earth 's surface , occasionally causing
strong winds that are commonly associated with thunderstorms . Thunderstorms can form and develop in any geographic location but most frequently within the
mid-latitude , where warm , moist air from tropical latitudes collides with cooler air from polar latitudes . Thunderstorms are responsible for the development
and formation of many severe weather phenomena . Thunderstorms , and the phenomena that occur along with them , pose great hazards . Damage that results from
thunderstorms is mainly inflicted by downburst winds , large hailstones , and flash flooding caused by heavy precipitation . Stronger thunderstorm cells are
capable of producing tornadoes and waterspouts . There are four types of thunderstorms : single - cell , multi-cell cluster , multi-cell lines , and supercells
.
</item 4>

<item 5>
Thunderstorm
Archived from the original on 2009 - 06 - 21 . Retrieved 2009 - 08 - 28 . ^ Jump up to : John E. Oliver ( 2005 ) . Encyclopedia of World Climatology . Springer
. p. 401 . ISBN 978 - 1 - 4020 - 3264 - 6 . Retrieved 2009 - 08 - 28 . Jump up ^ Dongxia Liu ; Guili Feng & Shujun Wu ( February 2009 ) . `` The characteristics
of cloud - to - ground lightning activity in hailstorms over northern China '' . Atmospheric Research . 91 ( 2 -- 4 ) : 459 -- 465 . Bibcode : 2009AtmRe ... 91
... 459L . doi : 10.1016 / j. atmosres. 2008.06. 016 . Jump up ^ Damir Počakal ; Željko Večenaj & Janez Štalec ( 2009 ) .
</item 5>

<item 6>
Thunderstorm
Motion ( edit ) Thunderstorm line viewed in reflectivity ( dBZ ) on a plan position indicator radar display The two major ways thunderstorms move are via
advection of the wind and propagation along outflow boundaries towards sources of greater heat and moisture . Many thunderstorms move with the mean wind speed
through the Earth 's troposphere , the lowest 8 kilometres ( 5.0 mi ) of the Earth 's atmosphere . Weaker thunderstorms are steered by winds closer to the Earth
's surface than stronger thunderstorms , as the weaker thunderstorms are not as tall . Organized , long - lived thunderstorm cells and complexes move at a right
angle to the direction of the vertical wind shear vector . If the gust front , or leading edge of the outflow boundary , races ahead of the thunderstorm , its
motion will accelerate in tandem . This is more of a factor with thunderstorms with heavy precipitation ( HP ) than with thunderstorms with low precipitation (
LP ) . When thunderstorms merge , which is most likely when numerous thunderstorms exist in proximity to each other , the motion of the stronger thunderstorm
normally dictates the future motion of the merged cell .
</item 6>

<item 7>
Thunderstorm
Radio pulses produced by cosmic rays are being used to study how electric charges develop within thunderstorms . More organized meteorological projects such as
VORTEX2 use an array of sensors , such as the Doppler on Wheels , vehicles with mounted automated weather stations , weather balloons , and unmanned aircraft to
investigate thunderstorms expected to produce severe weather . Lightning is detected remotely using sensors that detect cloud - to - ground lightning strokes
with 95 percent accuracy in detection and within 250 metres ( 820 ft ) of their point of origin . Mythology and religion ( edit ) Thunderstorms strongly
influenced many early civilizations . Greeks believed that they were battles waged by Zeus , who hurled lightning bolts forged by Hephaestus . Some American
Indian tribes associated thunderstorms with the Thunderbird , who they believed was a servant of the Great Spirit . The Norse considered thunderstorms to occur
when Thor went to fight Jötnar , with the thunder and lightning being the effect of his strikes with the hammer Mjölnir . Hinduism recognizes Indra as the god
of rain and thunderstorms .
</item 7>

<item 8>
Thunderstorm
Safety ( edit ) While safety and preparedness often overlap , `` thunderstorm safety '' generally refers to what people should do during and after a storm . The
American Red Cross recommends that people follow these precautions if a storm is imminent or in progress : Take action immediately upon hearing thunder . Anyone
close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless
telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe
or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid
touching metal . The NWS stopped recommending the `` lightning crouch '' in 2008 as it does n't provide a significant level of protection and will not
significantly lower the risk of being killed or injured from a nearby lightning strike . Frequent occurrences ( edit ) See also : United States rainfall
climatology A mild thunderstorm over Niagara Falls , Ontario .
</item 8>

<item 9>
Thunderstorm
Retrieved 2009 - 09 - 09 . Jump up ^ Charles H. Paxton ; J. Colson & N. Carlisle ( 2008 ) . `` P2. 13 Florida lightning deaths and injuries 2004 -- 2007 '' .
American Meteorological Society . Retrieved 2009 - 09 - 05 . Jump up ^ G.E. Likens ; W.C. Keene ; J.M. Miller & J.N. Galloway ( 1987 ) . `` Chemistry of
precipitation from a remote , terrestrial site in Australia '' . Journal of Geophysical Research . 92 ( 13 ) : 299 -- 314 . Bibcode : 1987JGR ... 92 ... 299R .
doi : 10.1029 / JA092iA01p00299 . Jump up ^ Joel S. Levine ; Tommy R. Augustsson ; Iris C. Andersont ; James M. Hoell Jr . & Dana A. Brewer ( 1984 ) . ``
Tropospheric sources of NOx : Lightning and biology '' . Atmospheric Environment.
</item 9>

<item 10>
Thunderstorm
doi : 10.1111 / cea. 12709 . PMID 26765082 ^ Jump up to : American Red Cross . `` Thunderstorm Safety Checklist '' ( PDF ) . American Red Cross . Retrieved 24
August 2011 . ^ Jump up to : National Weather Service Weather Forecast Office . `` Thunderstorm '' . Severe Weather Preparedness Information . Albuquerque , NM
: NOAA . Retrieved 24 August 2011 . Jump up ^ Federal Emergency Management Agency . `` Thunderstorms and Lightning '' . Ready . US Department of Homeland
Security . Archived from the original on 23 June 2011 . Retrieved 24 August 2011 . ^ Jump up to : Federal Emergency Management Agency . `` What to Do Before a
Thunderstorm '' . US Department of Homeland Security . Archived from the original on 20 August 2011 . Retrieved 24 August 2011 . ^ Jump up to : `` NWS Lightning
Safety Myths '' . Lightningsafety.noaa.gov. 2014 - 06 - 30 .
</item 10>
=== END ITEM id=2edab24496c1a9f7 ===
